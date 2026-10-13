"""Batch-spatialize a folder of mono source audio into binaural training clips."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from render import (HrirBank, room_rir, render_snapshots, adaptive_hrtf_nodes,
                    adaptive_position_nodes)


def stable_split(group):
    # Keep all samples from one speaker/book group in exactly one split.
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else ('validation' if bucket < 90 else 'test')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True, help='Recursive folder of mono WAV/FLAC/OGG')
    p.add_argument('--output-dir', type=Path, default=Path('dataset_out'))
    p.add_argument('--sofa', type=Path, default=Path(__file__).parent/'data/hrtf_nh2.sofa')
    p.add_argument('--mode', choices=['hrtf', 'room'], default='hrtf')
    p.add_argument('--variants', type=int, default=4, help='Spatial variants made for each source file')
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--step-ms', type=float, default=20,
                   help='Maximum time between HRTF snapshots; angular sampling also applies')
    p.add_argument('--max-angle-step-deg', type=float, default=1.0)
    p.add_argument('--rir-step-ms', type=float, default=100)
    p.add_argument('--max-rir-distance-m', type=float, default=0.2)
    p.add_argument('--absorption', type=float, default=0.4)
    p.add_argument('--max-order', type=int, default=3)
    p.add_argument('--room', nargs=3, type=float, default=[6, 5, 3])
    p.add_argument('--duration', type=float, default=30, help='Maximum output source duration')
    a = p.parse_args()
    if (a.variants < 1 or min(a.step_ms, a.rir_step_ms, a.duration,
                               a.max_angle_step_deg, a.max_rir_distance_m) <= 0):
        p.error('variants and time steps/duration must be positive')
    if not a.sofa.is_file():
        p.error(f'SOFA file not found: {a.sofa}')
    files = sorted(f for f in a.input_dir.rglob('*') if f.suffix.lower() in {'.wav', '.flac', '.ogg'})
    if not files:
        p.error('No WAV/FLAC/OGG input files found')
    bank = HrirBank(a.sofa)
    dims = np.array(a.room, dtype=float)
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
                for variant in range(a.variants):
                    rng = np.random.default_rng(a.seed + fi*100003 + variant)
                    start_side = 1 if variant % 2 == 0 else -1
                    angle0 = start_side*rng.uniform(20, 75)
                    angle1 = (-angle0 if variant % 4 != 3 else angle0)
                    elevation = rng.uniform(-10, 25)
                    radius = rng.uniform(1.2, min(3.5, np.min(dims[:2])*.4))
                    # Around a listener facing +x: azimuth positive means left.
                    to_xyz = lambda deg: listener + radius*np.array([
                        np.cos(np.deg2rad(deg))*np.cos(np.deg2rad(elevation)),
                        np.sin(np.deg2rad(deg))*np.cos(np.deg2rad(elevation)),
                        np.sin(np.deg2rad(elevation))])
                    start, end = to_xyz(angle0), to_xyz(angle1)
                    nodes, positions = adaptive_hrtf_nodes(start, end, listener, len(source), bank.fs,
                                                           a.max_angle_step_deg, a.step_ms)
                    if a.mode == 'room':
                        rir_nodes, rir_positions = adaptive_position_nodes(
                            start, end, len(source), bank.fs,
                            a.max_rir_distance_m, a.rir_step_ms)
                        rirs = [room_rir(pos, listener, dims, a.absorption, a.max_order, bank.fs)[None, :]
                                for pos in rir_positions]
                        room_signal = render_snapshots(source, rir_nodes, rirs)[:, 0]
                    else:
                        room_signal = source
                    hrir_filters = [bank.at(pos-listener)/np.linalg.norm(pos-listener) for pos in positions]
                    binaural = render_snapshots(room_signal, nodes, hrir_filters)
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
                    for t, pos in [(nodes[0], positions[0]), (nodes[-1], positions[-1])]:
                        v = pos-listener
                        labels.append({'dry_time_s': float(t/bank.fs),
                                       'xyz_m': pos.tolist(),
                                       'azimuth_deg': float(np.rad2deg(np.arctan2(v[1],v[0]))),
                                       'elevation_deg': float(np.rad2deg(np.arctan2(v[2],np.hypot(v[0],v[1])))),
                                       'distance_m': float(np.linalg.norm(v))})
                    row = {'id': key, 'audio': out.as_posix(), 'split': split,
                           'source_group': group, 'source_sha256': source_sha,
                           'source_path': rel.as_posix(), 'mode': a.mode,
                           'motion': 'static' if variant % 4 == 3 else ('left_to_right' if angle0 > 0 else 'right_to_left'),
                           'sample_rate': bank.fs, 'channels': ['left', 'right'],
                           'seed': a.seed + fi*100003 + variant, 'room_m': dims.tolist(),
                           'listener_m': listener.tolist(), 'rir_step_ms': a.rir_step_ms if a.mode == 'room' else None,
                           'max_rir_distance_m': a.max_rir_distance_m if a.mode == 'room' else None,
                           'rir_snapshot_count': len(rir_nodes) if a.mode == 'room' else None,
                           'hrtf_step_ms': a.step_ms, 'max_angle_step_deg': a.max_angle_step_deg,
                           'hrtf_snapshot_count': int(len(nodes)), 'global_gain': gain,
                           'trajectory_reference': 'dry input emission time; output has acoustic/filter latency',
                           'trajectory_interpolation': 'linear in XYZ between the two knots over dry input time',
                           'trajectory': labels}
                    manifest.write(json.dumps(row, ensure_ascii=False)+'\n')
                    made += 1
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
