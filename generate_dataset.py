"""Spatialize speech / sound-effect sources into labelled binaural scenes.

Input: one or more source manifests from prepare_sources.py (JSONL with audio,
type, text, group). Output: <output-dir>/<split>/<id>.wav, manifest.jsonl with
structured labels and template captions, errors.json.

  python generate_dataset.py --sources sources/speech.jsonl sources/sfx.jsonl \
      --output-dir generated --variants 2 --workers 16
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from captions import make_captions
from render import HrirBank
from scene import (ROOMS, Hrirs, distance_class, path_fraction, place_listener, positions,
                   render_scene, sample_motion, sample_room, sample_timing, sector_of, wrap_deg)

HRIRS = None


def stable_split(group):
    # Keep all samples from one speaker/video group in exactly one split.
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else ('validation' if bucket < 90 else 'test')


def allowed(source, room_kind, traj):
    """Content-specific rules (e.g. vehicles outdoors). Everything allowed for now."""
    return True


def load_source(path, fs, max_seconds, target_dbfs):
    """Mono, resampled, trimmed, active-RMS normalised.

    Returns (signal, active seconds, (first, last) sample of the active span).
    """
    data, rate = sf.read(path, dtype='float32', always_2d=True)
    if not np.isfinite(data).all():
        raise ValueError('audio contains NaN/Inf')
    x = data.mean(axis=1)  # AudioCaps is often stereo; spatial cues come from us
    divisor = np.gcd(rate, fs)
    x = resample_poly(x, fs//divisor, rate//divisor)[:round(max_seconds*fs)]
    frame = round(0.02*fs)
    frames = len(x)//frame
    if frames < 5:
        raise ValueError('clip too short')
    energy = np.mean(x[:frames*frame].reshape(frames, frame)**2, axis=1)
    if energy.max() <= 0:
        raise ValueError('silent clip')
    active = energy > energy.max()*1e-4  # within 40 dB of the loudest frame
    # Level is fixed before rendering so distance and room change loudness.
    gain = 10**(target_dbfs/20)/np.sqrt(np.mean(energy[active]))
    loud = np.flatnonzero(energy > energy.max()*1e-3)  # span for motion: within 30 dB
    return x*gain, active.sum()*frame/fs, (int(loud[0]*frame), int((loud[-1]+1)*frame))


def endpoint(listener, point):
    v = point-listener
    az = float(np.rad2deg(np.arctan2(v[1], v[0])))
    r = float(np.linalg.norm(v))
    return {'sector': sector_of(az), 'distance_class': distance_class(r), 'azimuth_deg': az,
            'elevation_deg': float(np.rad2deg(np.arctan2(v[2], np.hypot(v[0], v[1])))),
            'distance_m': r}


def describe(traj, listener, n, fs):
    p0, p1 = positions(traj, listener, [0, 1])
    spatial = {'motion': traj['kind'], 'start': endpoint(listener, p0), 'end': endpoint(listener, p1)}
    if traj['kind'] in ('arc', 'line'):
        sweep = traj['az1']-traj['az0']
        spatial['rotation'] = 'counterclockwise' if sweep > 0 else 'clockwise'  # seen from above
        spatial['sweep_deg'] = float(abs(sweep))
    if traj['kind'] == 'arc':
        spatial['via_sector'] = sector_of(wrap_deg((traj['az0']+traj['az1'])/2))
    if traj['kind'] == 'radial':
        spatial['radial'] = 'approach' if traj['r1'] < traj['r0'] else 'recede'
    if 'move' in traj:  # held at the start/end position outside this window
        spatial['move_start_s'] = round(traj['move'][0]/fs, 3)
        spatial['move_end_s'] = round(traj['move'][1]/fs, 3)
    # 10 Hz knots over dry (emission) time: [t, azimuth, elevation, distance].
    times = np.arange(0, n/fs, 0.1) if traj['kind'] != 'static' else np.zeros(1)
    pts = positions(traj, listener, path_fraction(traj, times*fs, n))-listener
    spatial['trajectory_10hz'] = [
        [round(float(t), 3), round(float(np.rad2deg(np.arctan2(v[1], v[0]))), 2),
         round(float(np.rad2deg(np.arctan2(v[2], np.hypot(v[0], v[1])))), 2),
         round(float(np.linalg.norm(v)), 3)] for t, v in zip(times, pts)]
    dense = positions(traj, listener, np.linspace(0, 1, 256))-listener
    spatial['min_distance_m'] = float(np.min(np.linalg.norm(dense, axis=1)))
    return spatial


def sample_scene(rng, source):
    for _ in range(200):
        room = sample_room(rng, str(rng.choice(list(ROOMS))))
        for _ in range(20):
            traj = sample_motion(rng, room['kind'])
            if not allowed(source, room['kind'], traj):
                continue
            listener = place_listener(rng, room, traj)
            if listener is not None:
                return room, traj, listener
    raise ValueError('could not place a scene')


def init_worker(sofa):
    global HRIRS
    HRIRS = Hrirs(HrirBank(sofa))


def process(job):
    """Render every variant of one source; returns (rows, error or None)."""
    source, base, a = job['source'], job['base'], job['args']
    fs = HRIRS.fs
    try:
        signal, active, span = load_source(source['audio'], fs, a['max_duration'], a['target_dbfs'])
        if active < a['min_active']:
            raise ValueError(f'only {active:.2f}s active audio')
        rows = []
        for variant in job['variants']:
            key = f'{base}_v{variant:02d}'
            seed = [a['seed'], int(base, 16), variant]
            rng = np.random.default_rng(seed)
            room, traj, listener = sample_scene(rng, source)
            traj = sample_timing(rng, traj, span, fs)
            audio, nodes, _ = render_scene(signal, HRIRS, traj, listener, room, rng)
            peak = float(np.max(np.abs(audio)))
            clip_gain = min(1.0, 0.99/max(peak, 1e-12))  # only touches rare overloads
            audio *= clip_gain
            if a['out_sr'] != fs:
                divisor = np.gcd(a['out_sr'], fs)
                audio = resample_poly(audio, a['out_sr']//divisor, fs//divisor, axis=0)
            split = stable_split(source['group'])
            out = Path(a['output_dir'])/split/f'{key}.wav'
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, audio, a['out_sr'], subtype='PCM_24')
            spatial = describe(traj, listener, len(signal), fs)
            rows.append({
                'id': key, 'audio': f'{split}/{key}.wav', 'split': split,
                'sample_rate': a['out_sr'], 'channels': ['left', 'right'],
                'duration_s': round(len(audio)/a['out_sr'], 3),
                'dry_duration_s': round(len(signal)/fs, 3),  # rest is propagation + reverb tail
                'source':{k: source.get(k) for k in
                           ('type', 'text', 'group', 'audio', 'source_dataset', 'source_id', 'language')},
                'room': {'kind': room['kind'],
                         'dims_m': None if room['dims'] is None else room['dims'].round(2).tolist(),
                         'rt60_s': room['rt60'], 'listener_m': listener.round(3).tolist()},
                'spatial': spatial,
                'captions': make_captions(source, room['kind'], spatial),
                'render': {'engine': 'scene.py: HRIR image sources (order 2) + diffuse tail',
                           'hrtf_snapshot_count': int(len(nodes)), 'source_dbfs': a['target_dbfs'],
                           'clip_gain': clip_gain, 'seed': seed},
            })
        return rows, None
    except Exception as exc:  # one bad file must not stop a large run
        return [], {'audio': source.get('audio'), 'error': f'{type(exc).__name__}: {exc}'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', type=Path, nargs='+', required=True, help='Source manifests (JSONL)')
    p.add_argument('--output-dir', type=Path, default=Path('generated'))
    p.add_argument('--sofa', type=Path, default=Path(__file__).parent/'data/hrtf_nh2.sofa')
    p.add_argument('--variants', type=int, default=2, help='Spatial scenes per source clip')
    p.add_argument('--seed', type=int, default=20261017)
    p.add_argument('--max-duration', type=float, default=15.0, help='Trim sources to this many seconds')
    p.add_argument('--min-active', type=float, default=1.0, help='Minimum non-silent seconds')
    p.add_argument('--target-dbfs', type=float, default=-30.0,
                   help='Active RMS of the dry source; leaves headroom for near sources in small rooms')
    p.add_argument('--out-sr', type=int, default=48000)
    p.add_argument('--limit', type=int, default=None, help='Use only the first N sources (smoke tests)')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2)-1))
    p.add_argument('--resume', action='store_true', help='Skip ids already in manifest.jsonl')
    a = p.parse_args()
    if a.variants < 1 or a.max_duration <= 0 or a.out_sr <= 0:
        p.error('variants, max-duration and out-sr must be positive')
    if not a.sofa.is_file():
        p.error(f'SOFA file not found: {a.sofa}')
    sources = []
    for manifest in a.sources:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                path = Path(row['audio'])
                row['audio'] = str(path if path.is_absolute() else (manifest.parent/path).resolve())
                sources.append(row)
    sources = sources[:a.limit]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output_dir/'manifest.jsonl'
    done = set()
    if a.resume and manifest_path.exists():
        done = {json.loads(l)['id'] for l in manifest_path.read_text(encoding='utf-8').splitlines() if l}
    shared = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != 'sources'}
    jobs = []
    for source in sources:
        # Id (and seed) from the source identity, so adding files never
        # reshuffles others and moving the data keeps ids stable.
        identity = (f"{source['source_dataset']}:{source['source_id']}" if source.get('source_id')
                    else source['audio'])
        base = hashlib.sha256(identity.encode()).hexdigest()[:16]
        variants = [v for v in range(a.variants) if f'{base}_v{v:02d}' not in done]
        if variants:
            jobs.append({'source': source, 'base': base, 'variants': variants, 'args': shared})
    print(f'{len(sources)} sources, {len(jobs)} to render, {len(done)} clips already done', flush=True)
    made, errors = 0, []
    with manifest_path.open('a' if a.resume else 'w', encoding='utf-8') as manifest, \
            ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(str(a.sofa),)) as pool:
        for i, (rows, error) in enumerate(pool.map(process, jobs, chunksize=4)):
            for row in rows:
                manifest.write(json.dumps(row, ensure_ascii=False)+'\n')
            made += len(rows)
            if error:
                errors.append(error)
            if (i+1) % 50 == 0 or i+1 == len(jobs):
                manifest.flush()
                print(f'[{i+1}/{len(jobs)}] {made} clips, {len(errors)} errors', flush=True)
    (a.output_dir/'errors.json').write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Done: {made} clips; {len(errors)} source errors; manifest={manifest_path}')


if __name__ == '__main__':
    main()
