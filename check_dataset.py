"""Automatic sanity checks on a generated dataset (not a perceptual test).

- Lateral sources (30-150 deg off the median plane): the interaural time
  difference (GCC-PHAT) and the level difference must point to the labelled
  side, at the start and at the end of each clip. GCC-PHAT locks onto the
  coherent direct sound; the diffuse tail is incoherent between the ears.
  Broadband ILD is reported too, but in rooms beyond the critical distance it
  is weak and fluctuates by a dB or two, as in real rooms.
- Level per room and distance class.
- Actual shares of scene, motion, distance and sector vs ratios.json.

  python check_dataset.py generated/manifest.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def itd_s(x, fs, band=(200, 8000)):
    """GCC-PHAT lag in seconds; positive = right ear later = source on the left.

    Band-limited: PHAT gives every bin equal weight, and bins with no signal
    (above the HRIR/MP3 bandwidth) hold rounding noise that is identical in
    both ears and would produce a false zero-lag peak.
    """
    n = 1 << int(np.ceil(np.log2(2*len(x))))
    cross = np.fft.rfft(x[:, 1], n)*np.conj(np.fft.rfft(x[:, 0], n))
    freq = np.fft.rfftfreq(n, 1/fs)
    cross = np.where((freq >= band[0]) & (freq <= band[1]), cross/(np.abs(cross)+1e-20), 0)
    cc = np.fft.irfft(cross, n)
    max_lag = int(0.001*fs)
    cc = np.r_[cc[-max_lag:], cc[:max_lag+1]]
    return (np.argmax(cc)-max_lag)/fs


def ild_db(x):
    return 10*np.log10((np.mean(x[:, 0]**2)+1e-15)/(np.mean(x[:, 1]**2)+1e-15))


def report_shares(rows, ratios):
    """Actual share of each attribute next to the configured ratio."""
    def motion(r):
        return r['spatial'].get('radial', r['spatial']['motion'])
    groups = {
        'scene': [r['room']['kind'] for r in rows],
        'motion': [motion(r) for r in rows],
        'distance': [r['spatial']['start']['distance_class'] for r in rows
                     if r['spatial']['motion'] != 'radial'],
        'sector': [r['spatial']['start']['sector'] for r in rows],
    }
    print('Shares (actual / configured):')
    for group, values in groups.items():
        target = ratios.get(group, {}) if ratios else {}
        total = sum(target.values()) or 1
        counts = defaultdict(int)
        for v in values:
            counts[v] += 1
        cells = [f'{k} {counts[k]/max(len(values), 1):.3f}/{target.get(k, 0)/total:.3f}'
                 for k in sorted(set(counts) | set(target))]
        print(f'  {group:8s} ' + '  '.join(cells))
    released = sum(1 for r in rows if r.get('plan', {}).get('distance') not in (None,
                   r['spatial']['start']['distance_class']))
    print(f'  planned distance not placeable (rendered at another distance): {released}/{len(rows)}')


def main():
    manifest = Path(sys.argv[1])
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line]
    ratios_path = manifest.parent/'ratios.json'
    report_shares(rows, json.loads(ratios_path.read_text(encoding='utf-8')) if ratios_path.exists() else None)
    ok = defaultdict(lambda: [0, 0, 0])  # itd ok, ild ok, total
    levels = defaultdict(list)
    bad = []
    for row in rows:
        audio, fs = sf.read(manifest.parent/row['audio'])
        room, spatial = row['room']['kind'], row['spatial']
        # First and last second of activity within the dry span; silence (and
        # the reverb tail after the source stops) carries no direction.
        dry = round(row['dry_duration_s']*fs)
        frame = int(0.02*fs)
        e = np.mean(audio[:dry//frame*frame].reshape(-1, frame, 2)**2, axis=(1, 2))
        active = np.flatnonzero(e > e.max()*0.01)*frame  # within 20 dB of the peak
        if len(active) < 10:
            continue
        first, last = active[0], active[-1]+frame
        second = min((last-first)//2, fs)
        for name, part in (('start', audio[first:first+second]), ('end', audio[last-second:last])):
            az = spatial[name]['azimuth_deg']
            if 30 <= abs(az) <= 150 and np.mean(part**2) > 1e-10:
                key = (room, spatial['motion'])
                itd, ild = itd_s(part, fs), ild_db(part)
                ok[key][0] += np.sign(itd) == np.sign(az)
                ok[key][1] += np.sign(ild) == np.sign(az)
                ok[key][2] += 1
                if np.sign(itd) != np.sign(az):
                    bad.append((row['id'], name, round(az), round(itd*1e6), round(ild, 1)))
        if spatial['motion'] == 'static':
            levels[(room, spatial['start']['distance_class'])].append(
                10*np.log10(np.mean(audio**2)+1e-15))
    print('Lateral endpoints pointing to the labelled side:  ITD (GCC-PHAT) | broadband ILD')
    for key in sorted(ok):
        itd_ok, ild_ok, total = ok[key]
        print(f'  {key[0]:7s} {key[1]:7s} {itd_ok:3d}/{total:<3d} | {ild_ok:3d}/{total}')
    print('Static clips, mean level (dB):')
    for key in sorted(levels):
        print(f'  {key[0]:7s} {key[1]:5s} {np.mean(levels[key]):6.1f}  (n={len(levels[key])})')
    if bad:
        print('ITD mismatches (id, endpoint, azimuth, ITD us, ILD dB):')
        for item in bad[:20]:
            print('  ', *item)


if __name__ == '__main__':
    main()
