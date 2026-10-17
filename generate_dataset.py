"""Batch-spatialize a folder of mono source audio into binaural training clips."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from render import HrirBank, closest_distance, render_binaural

MIN_RADIUS = 1.2  # ARI HRIRs were measured at 1.2 m; no near-field model.


def stable_split(group):
    # Keep all samples from one speaker/book group in exactly one split.
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else ('validation' if bucket < 90 else 'test')


def sample_trajectory(rng, listener, dims, static, min_distance=1.0, margin=0.1, max_radius=3.5):
    """Draw a static point or a straight chord around the listener.

    Azimuth covers the full circle (front and rear, left and right). Chords are
    rejected if they leave the room or pass closer than min_distance.
    """
    r_hi = min(max_radius, float(np.min(dims[:2]))*.4)
    for _ in range(1000):
        radius = rng.uniform(MIN_RADIUS, r_hi)
        elevation = np.deg2rad(rng.uniform(-10, 25))
        centre = rng.uniform(-180, 180)
        # Around a listener facing +x: azimuth positive means left.
        to_xyz = lambda deg: listener + radius*np.array([
            np.cos(np.deg2rad(deg))*np.cos(elevation),
            np.sin(np.deg2rad(deg))*np.cos(elevation),
            np.sin(elevation)])
        if static:
            start = end = to_xyz(centre)
        else:
            half = rng.uniform(20, 75)
            sign = rng.choice([-1, 1])
            start, end = to_xyz(centre+sign*half), to_xyz(centre-sign*half)
        # Box is convex, so checking both endpoints covers the whole segment.
        inside = all(np.all((v > margin) & (v < dims-margin)) for v in (start, end))
        if inside and closest_distance(start, end, listener) >= min_distance:
            return start, end
    raise ValueError('could not place a trajectory inside the room; enlarge --room or lower --min-distance')


def azimuth_deg(v):
    return float(np.rad2deg(np.arctan2(v[1], v[0])))


def motion_label(start, end, listener):
    if np.allclose(start, end):
        return 'static'
    a0, a1 = azimuth_deg(start-listener), azimuth_deg(end-listener)
    # Sweeps are < 180 deg, so the wrapped difference gives the direction.
    # Decreasing azimuth = clockwise seen from above (left-to-right when in front).
    return 'azimuth_decreasing' if (a1-a0+180) % 360-180 < 0 else 'azimuth_increasing'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True, help='Recursive folder of mono WAV/FLAC/OGG')
    p.add_argument('--output-dir', type=Path, default=Path('dataset_out'))
    p.add_argument('--sofa', type=Path, default=Path(__file__).parent/'data/hrtf_nh2.sofa')
    p.add_argument('--mode', choices=['hrtf', 'room'], default='hrtf')
    p.add_argument('--variants', type=int, default=4,
                   help='Spatial variants made for each source file; every 4th is static')
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--step-ms', type=float, default=20,
                   help='Maximum time between HRTF snapshots; angular sampling also applies')
    p.add_argument('--max-angle-step-deg', type=float, default=1.0)
    p.add_argument('--rir-step-ms', type=float, default=100)
    p.add_argument('--max-rir-distance-m', type=float, default=0.2)
    p.add_argument('--max-rir-radial-m', type=float, default=None,
                   help='Optional bound on source-listener distance change between RIR snapshots')
    p.add_argument('--min-distance', type=float, default=1.0,
                   help='Minimum source-listener distance along every trajectory (m)')
    p.add_argument('--absorption', type=float, default=0.4)
    p.add_argument('--max-order', type=int, default=3)
    p.add_argument('--room', nargs=3, type=float, default=[6, 5, 3])
    p.add_argument('--duration', type=float, default=30, help='Maximum output source duration')
    a = p.parse_args()
    if (a.variants < 1 or min(a.step_ms, a.rir_step_ms, a.duration, a.min_distance,
                               a.max_angle_step_deg, a.max_rir_distance_m) <= 0 or
            (a.max_rir_radial_m is not None and a.max_rir_radial_m <= 0)):
        p.error('variants, time steps, distances and duration must be positive')
    if not a.sofa.is_file():
        p.error(f'SOFA file not found: {a.sofa}')
    dims = np.array(a.room, dtype=float)
    if min(3.5, float(np.min(dims[:2]))*.4) < MIN_RADIUS:
        p.error(f'room too small: needs min(length, width) >= {MIN_RADIUS/.4:.1f} m')
    files = sorted(f for f in a.input_dir.rglob('*') if f.suffix.lower() in {'.wav', '.flac', '.ogg'})
    if not files:
        p.error('No WAV/FLAC/OGG input files found')
    bank = HrirBank(a.sofa)
    listener = dims*np.array([.35, .5, .5])
    manifest_path = a.output_dir/'manifest.jsonl'
    a.output_dir.mkdir(parents=True, exist_ok=True)
    made, errors = 0, []
    with manifest_path.open('w', encoding='utf-8') as manifest:
        for fi, path in enumerate(files):
            try:
                data, fs = sf.read(path, dtype='float32', always_2d=True)
                if data.shape[1] != 1:
                    raise ValueError(f'expected mono, got {data.shape[1]} channels')
                if not np.isfinite(data).all():
                    raise ValueError('audio contains NaN/Inf')
                divisor = np.gcd(fs, bank.fs)
                source = resample_poly(data[:, 0], bank.fs//divisor, fs//divisor)
                source = source[:round(a.duration*bank.fs)]
                if len(source) < bank.fs//2:
                    raise ValueError('clip shorter than 0.5 s')
                # For LibriSpeech-style trees, the first component below the input root is speaker ID.
                rel = path.relative_to(a.input_dir)
                group = rel.parts[0] if len(rel.parts) > 1 else path.stem
                split = stable_split(group)
                source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                # Seed from the relative path, so adding files does not reshuffle others.
                path_seed = int(hashlib.sha256(rel.as_posix().encode()).hexdigest()[:8], 16)
                rows = []
                for variant in range(a.variants):
                    seed = [a.seed, path_seed, variant]
                    rng = np.random.default_rng(seed)
                    start, end = sample_trajectory(rng, listener, dims, variant % 4 == 3, a.min_distance)
                    binaural, nodes, _, rir_count = render_binaural(
                        source, bank, start, end, listener, a.mode, dims, a.absorption,
                        a.max_order, a.step_ms, a.max_angle_step_deg, a.rir_step_ms,
                        a.max_rir_distance_m, a.max_rir_radial_m)
                    peak = float(np.max(np.abs(binaural)))
                    gain = min(1.0, .95/max(peak, 1e-12))
                    binaural *= gain
                    key = f'{source_sha[:16]}_v{variant:03d}'
                    out = a.output_dir/split/f'{key}.wav'
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, binaural, bank.fs, subtype='PCM_24')
                    # The trajectory is a straight line in XYZ, so store its
                    # knots compactly. Training code can interpolate labels at
                    # any feature-frame rate without bloating the manifest.
                    labels = []
                    for t, pos in [(0, start), (len(source)-1, end)]:
                        v = pos-listener
                        labels.append({'dry_time_s': float(t/bank.fs),
                                       'xyz_m': pos.tolist(),
                                       'azimuth_deg': azimuth_deg(v),
                                       'elevation_deg': float(np.rad2deg(np.arctan2(v[2],np.hypot(v[0],v[1])))),
                                       'distance_m': float(np.linalg.norm(v))})
                    rows.append({'id': key, 'audio': f'{split}/{key}.wav', 'split': split,
                                 'source_group': group, 'source_sha256': source_sha,
                                 'source_path': rel.as_posix(), 'mode': a.mode,
                                 'motion': motion_label(start, end, listener),
                                 'min_distance_m': closest_distance(start, end, listener),
                                 'sample_rate': bank.fs, 'channels': ['left', 'right'],
                                 'seed': seed, 'room_m': dims.tolist(),
                                 'listener_m': listener.tolist(), 'rir_step_ms': a.rir_step_ms if a.mode == 'room' else None,
                                 'max_rir_distance_m': a.max_rir_distance_m if a.mode == 'room' else None,
                                 'max_rir_radial_m': a.max_rir_radial_m if a.mode == 'room' else None,
                                 'rir_snapshot_count': rir_count,
                                 'hrtf_step_ms': a.step_ms, 'max_angle_step_deg': a.max_angle_step_deg,
                                 'hrtf_snapshot_count': int(len(nodes)), 'global_gain': gain,
                                 'trajectory_reference': 'dry input emission time; output has acoustic/filter latency',
                                 'trajectory_interpolation': 'linear in XYZ between the two knots over dry input time',
                                 'trajectory': labels})
                # Write rows only once every variant of this file succeeded.
                for row in rows:
                    manifest.write(json.dumps(row, ensure_ascii=False)+'\n')
                made += len(rows)
                print(f'[{fi+1}/{len(files)}] {path.name}: {a.variants} variants', flush=True)
            except Exception as exc:
                errors.append({'file': str(path), 'error': str(exc)})
                print(f'SKIP {path}: {exc}', flush=True)
    (a.output_dir/'errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
    print(f'Done: {made} clips; {len(errors)} input errors; manifest={manifest_path}')
    if not made:
        raise SystemExit('No output clips generated')


if __name__ == '__main__':
    main()
