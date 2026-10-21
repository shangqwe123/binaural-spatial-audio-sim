"""Educational RIR + HRIR cascade; not the authors' implementation.

World/listener axes: +x front, +y left, +z up. SOFA receiver 0=L, 1=R.
Each input sample is distributed between adjacent snapshot filters using
linear weights. Full convolution tails are retained (input-time crossfade).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import urllib.request

import h5py
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, correlate, correlation_lags
from scipy.spatial import cKDTree

ARI_URL = 'https://sofacoustics.org/data/database/ari/hrtf_nh2.sofa'


def attr_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


class HrirBank:
    """Restricted SOFA loader: spherical, listener-relative, zero Data.Delay.

    ARI example satisfies these constraints. Reject unsupported files instead
    of silently ignoring timing information or orientation transforms.
    """
    def __init__(self, path):
        with h5py.File(path, 'r') as f:
            self.ir = np.asarray(f['Data.IR'], dtype=float)
            pos = np.asarray(f['SourcePosition'], dtype=float)
            rates = np.asarray(f['Data.SamplingRate']).ravel()
            if not np.allclose(rates, rates[0]):
                raise ValueError('Multiple sampling rates are unsupported')
            self.fs = int(round(float(rates[0])))
            if attr_text(f['SourcePosition'].attrs.get('Type', '')).lower() != 'spherical':
                raise ValueError('SourcePosition must be spherical')
            if 'degree' not in attr_text(f['SourcePosition'].attrs.get('Units', '')).lower():
                raise ValueError('SourcePosition angles must be in degrees')
            if 'Data.Delay' in f and np.any(np.asarray(f['Data.Delay']) != 0):
                raise ValueError('Nonzero SOFA Data.Delay unsupported; bake delays into IR first')
            for key, expected in [('ListenerPosition', [0, 0, 0]),
                                  ('ListenerView', [1, 0, 0]),
                                  ('ListenerUp', [0, 0, 1])]:
                if key in f and not np.allclose(np.asarray(f[key]), expected):
                    raise ValueError(f'Unsupported {key}; requires default listener coordinates')
        if self.ir.ndim != 3 or self.ir.shape[1] != 2:
            raise ValueError('Expected Data.IR shape (measurements, 2, samples)')
        if len(pos) != len(self.ir) or not np.isfinite(self.ir).all():
            raise ValueError('Invalid SOFA data')
        az, el = np.deg2rad(pos[:, 0]), np.deg2rad(pos[:, 1])
        self.directions = np.c_[np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)]
        self.direction_tree = cKDTree(self.directions)
        # Separate approximate arrival delays from residual filter shape before
        # blending. Zero padding prevents circular wrap into the useful output.
        self.nfft = 4 * self.ir.shape[-1]
        self.freq = np.fft.rfftfreq(self.nfft)
        # Onset (first sample within -20 dB of the IR peak) is more stable than
        # the raw peak on the contralateral ear, where the peak is weak.
        magnitude = np.abs(self.ir)
        self.delays = np.argmax(magnitude >= 0.1*magnitude.max(axis=-1, keepdims=True),
                                axis=-1).astype(float)
        spectrum = np.fft.rfft(self.ir, n=self.nfft, axis=-1)
        self.aligned = spectrum * np.exp(
            2j*np.pi*self.delays[..., None]*self.freq)
        self.magnitude = np.abs(self.aligned)

    def at(self, direction, neighbors=12, sigma_deg=3.0):
        # Local spherical kernel (ARI grid spacing is ~5 deg) blends a few
        # measured neighbours. Magnitude and phase are blended separately:
        # averaging complex spectra cancels high frequencies, whose residual
        # phase differs between neighbours (that cost ~7 dB at 6-10 kHz and
        # ~14 dB at 10-16 kHz with the old 48-neighbour, 7.5-deg kernel).
        direction = direction / np.linalg.norm(direction)
        distances, indices = self.direction_tree.query(direction, k=min(neighbors, len(self.directions)))
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        angles = 2*np.arcsin(np.clip(distances/2, 0, 1))
        exponent = -0.5*(angles/np.deg2rad(sigma_deg))**2
        weights = np.exp(exponent-exponent.max())
        weights /= weights.sum()
        delays = weights @ self.delays[indices]
        magnitude = np.einsum('m,mcf->cf', weights, self.magnitude[indices], optimize=False)
        blended = np.einsum('m,mcf->cf', weights, self.aligned[indices], optimize=False)
        phase = blended/np.maximum(np.abs(blended), 1e-12)
        spectrum = magnitude*phase*np.exp(-2j*np.pi*delays[:, None]*self.freq)
        return np.fft.irfft(spectrum, n=self.nfft, axis=-1)[:, :2*self.ir.shape[-1]]


def adaptive_hrtf_nodes(start, end, listener, signal_length, fs,
                        max_angle_deg=1.0, max_step_ms=20.0):
    """Return source-motion knots bounded by angular and temporal spacing."""
    if signal_length < 2:
        return np.array([0]), np.asarray(start, dtype=float)[None, :]
    start, end, listener = map(lambda x: np.asarray(x, dtype=float), (start, end, listener))
    if np.allclose(start, end):
        return np.array([0]), start[None, :]
    delta = end-start

    def position(sample):
        return start + delta*(sample/(signal_length-1))

    def unit_direction(sample):
        vector = position(sample)-listener
        return vector/np.linalg.norm(vector)

    max_angle = np.deg2rad(max_angle_deg)
    max_samples = max(1, round(fs*max_step_ms/1000))
    nodes = [0]
    left = 0
    while left < signal_length-1:
        right = min(signal_length-1, left+max_samples)
        while True:
            # Check midpoint too, protecting against paths that swing around
            # the listener even when endpoint directions look deceptively close.
            mid = (left+right)//2
            dl, dr = unit_direction(left), unit_direction(right)
            angular = np.arccos(np.clip(np.dot(dl, dr), -1, 1))
            if mid > left and mid < right:
                dm = unit_direction(mid)
                angular = max(angular,
                              np.arccos(np.clip(np.dot(dl, dm), -1, 1)),
                              np.arccos(np.clip(np.dot(dm, dr), -1, 1)))
            if angular <= max_angle or right-left <= 1:
                break
            right = mid
        nodes.append(right)
        left = right
    nodes = np.asarray(nodes, dtype=int)
    return nodes, np.stack([position(i) for i in nodes])


def adaptive_position_nodes(start, end, signal_length, fs,
                            max_distance_m=0.2, max_step_ms=100.0,
                            listener=None, max_radial_m=None):
    """Bound source displacement and elapsed time between spatial snapshots.

    With listener and max_radial_m, also bound the change of source-listener
    distance. Crossfading two RIRs whose direct paths differ by d metres sums
    copies offset by d/c seconds (comb filtering), so this is the cheap bound
    that matters: tangential motion adds almost no snapshots.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if signal_length < 2 or np.allclose(start, end):
        return np.array([0]), start[None, :]
    delta = end-start
    distance = float(np.linalg.norm(delta))
    by_distance = max(1, int(np.ceil(distance/max_distance_m)))
    by_time = max(1, int(np.ceil((signal_length-1)/(fs*max_step_ms/1000))))
    count = max(by_distance, by_time)
    if listener is not None and max_radial_m:
        listener = np.asarray(listener, dtype=float)
        r0, r1 = np.linalg.norm(start-listener), np.linalg.norm(end-listener)
        r_min = closest_distance(start, end, listener)
        # |source-listener| is convex along a line: total variation is exact.
        count = max(count, int(np.ceil((r0+r1-2*r_min)/max_radial_m)))
    nodes = np.unique(np.rint(np.linspace(0, signal_length-1, count+1)).astype(int))
    positions = start[None, :] + delta[None, :]*nodes[:, None]/(signal_length-1)
    return nodes, positions


def closest_distance(start, end, listener):
    """Smallest source-listener distance along the straight segment."""
    start, end, listener = (np.asarray(v, dtype=float) for v in (start, end, listener))
    delta = end-start
    u = np.clip(np.dot(listener-start, delta)/max(np.dot(delta, delta), 1e-15), 0, 1)
    return float(np.linalg.norm(start+u*delta-listener))


def room_rir(source, listener, room_dims, absorption, order, fs):
    import pyroomacoustics as pra
    room = pra.ShoeBox(room_dims, fs=fs, materials=pra.Material(absorption), max_order=order)
    room.add_source(source)
    # Head-centre omnidirectional receiver; HRIRs add the binaural cues.
    room.add_microphone_array(listener[:, None])
    room.compute_rir()
    return np.asarray(room.rir[0][0])


def render_binaural(signal, bank, start, end, listener, mode, room_dims=None,
                    absorption=0.4, order=3, step_ms=20.0, max_angle_deg=1.0,
                    rir_step_ms=100.0, max_rir_distance_m=0.2, max_rir_radial_m=0.02,
                    progress=False):
    """Render a straight-line source; returns (audio, hrtf nodes, positions, RIR count)."""
    start, end, listener = (np.asarray(v, dtype=float) for v in (start, end, listener))
    nodes, positions = adaptive_hrtf_nodes(start, end, listener, len(signal), bank.fs,
                                           max_angle_deg, step_ms)
    if mode == 'hrtf':
        # Direction-only demo: relative 1/r gain, no additional flight-time delay.
        filters = [bank.at(pos-listener) / np.linalg.norm(pos-listener) for pos in positions]
        return render_snapshots(signal, nodes, filters), nodes, positions, None
    # Room geometry/absorption move slowly: simulate sparse RIR knots first,
    # then render at the fine HRTF rate. This avoids thousands of room builds.
    rir_nodes, rir_positions = adaptive_position_nodes(
        start, end, len(signal), bank.fs, max_rir_distance_m, rir_step_ms,
        listener, max_rir_radial_m)
    rirs = []
    for i, pos in enumerate(rir_positions):
        rirs.append(room_rir(pos, listener, room_dims, absorption, order, bank.fs)[None, :])
        if progress and i % 20 == 0:
            print(f'Room RIRs: {i+1}/{len(rir_nodes)}', flush=True)
    room_signal = render_snapshots(signal, rir_nodes, rirs)[:, 0]
    # The RIR already contains 1/(4*pi*r) spreading and flight time; HRIRs add
    # only direction here, otherwise distance attenuation is applied twice.
    filters = [bank.at(pos-listener) for pos in positions]
    return render_snapshots(room_signal, nodes, filters), nodes, positions, len(rirs)


def render_snapshots(signal, nodes, filters):
    """Piecewise-linear filter interpolation, with exact partition of unity."""
    channels = filters[0].shape[0]
    if any(h.ndim != 2 or h.shape[0] != channels for h in filters):
        raise ValueError('All filters must have shape (same channels, taps)')
    output = np.zeros((len(signal) + max(h.shape[1] for h in filters) - 1, channels))
    for i, (node, filt) in enumerate(zip(nodes, filters)):
        start = int(nodes[i-1]) if i else 0
        stop = int(nodes[i+1]) + 1 if i+1 < len(nodes) else len(signal)
        samples = np.arange(start, stop)
        weight = np.ones(len(samples))
        if i:
            left = samples < node
            weight[left] = (samples[left] - nodes[i-1]) / (node - nodes[i-1])
        if i+1 < len(nodes):
            right = samples > node
            weight[right] = (nodes[i+1] - samples[right]) / (nodes[i+1] - node)
        chunk = signal[start:stop] * weight
        for channel in range(channels):
            wet = fftconvolve(chunk, filt[channel])
            output[start:start+len(wet), channel] += wet
    return output


def cue_rows(audio, fs):
    """Broadband diagnostics, not perceptual quality metrics.

    Positive ITD means right arrives later; positive ILD means left is stronger.
    """
    length = max(2, round(fs * 0.05))
    for start in range(0, len(audio)-length+1, length):
        left, right = audio[start:start+length].T
        el, er = np.mean(left**2), np.mean(right**2)
        if max(el, er) < 1e-10:
            continue
        corr = correlate(right, left, mode='full', method='fft')
        lags = correlation_lags(len(right), len(left))
        valid = np.abs(lags) <= round(fs * 0.001)
        lag = lags[valid][np.argmax(corr[valid])]
        yield [(start+length/2)/fs, lag/fs*1e6, 10*np.log10((el+1e-15)/(er+1e-15))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, help='Mono dry audio; omit for seeded noise demo')
    p.add_argument('--sofa', type=Path, default=Path(__file__).parent/'data/hrtf_nh2.sofa')
    p.add_argument('--download-hrtf', action='store_true')
    p.add_argument('--output', type=Path, default=Path('outputs/left_to_right.wav'))
    p.add_argument('--mode', choices=['hrtf', 'room'], default='hrtf')
    p.add_argument('--seconds', type=float, default=5, help='Synthetic demo duration')
    p.add_argument('--start', nargs=3, type=float, default=[4, 4, 1.5])
    p.add_argument('--end', nargs=3, type=float, default=[4, 1, 1.5])
    p.add_argument('--listener', nargs=3, type=float, default=[2, 2.5, 1.5])
    p.add_argument('--room', nargs=3, type=float, default=[6, 5, 3])
    p.add_argument('--absorption', type=float, default=0.4)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--step-ms', type=float, default=20,
                   help='Maximum time between HRTF snapshots (adaptive angular sampling also applies)')
    p.add_argument('--max-angle-step-deg', type=float, default=1.0,
                   help='Maximum source angular change between HRTF snapshots')
    p.add_argument('--rir-step-ms', type=float, default=100,
                   help='Maximum time between RIR snapshots')
    p.add_argument('--max-rir-distance-m', type=float, default=0.2,
                   help='Maximum source displacement between RIR snapshots')
    p.add_argument('--max-rir-radial-m', type=float, default=0.02,
                   help='Maximum change of source-listener distance between RIR snapshots; '
                        'reduces comb filtering for approaching sources (0 disables)')
    a = p.parse_args()
    if (a.seconds <= 0 or a.step_ms <= 0 or a.rir_step_ms <= 0 or
            a.max_angle_step_deg <= 0 or a.max_rir_distance_m <= 0 or
            a.max_rir_radial_m < 0 or
            a.order < 0 or not 0 <= a.absorption <= 1):
        p.error('Invalid duration, step, reflection order or absorption')
    if not a.sofa.exists() and a.download_hrtf:
        a.sofa.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.sofa.with_suffix('.download')
        with urllib.request.urlopen(ARI_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(a.sofa)
    if not a.sofa.exists():
        p.error('SOFA missing: pass --download-hrtf or --sofa PATH')
    bank = HrirBank(a.sofa)
    if a.input:
        signal, fs = sf.read(a.input, always_2d=True)
        if signal.shape[1] != 1:
            p.error('Use mono dry input; stereo is not silently downmixed')
        signal = signal[:, 0]
        divisor = math.gcd(fs, bank.fs)
        signal = resample_poly(signal, bank.fs//divisor, fs//divisor)
    else:
        signal = np.random.default_rng(42).normal(0, 0.08, round(a.seconds*bank.fs))
        fade = min(round(0.03*bank.fs), len(signal)//2)
        if fade:
            signal[:fade] *= np.linspace(0, 1, fade)
            signal[-fade:] *= np.linspace(1, 0, fade)
    if len(signal) < 2 or not np.isfinite(signal).all():
        p.error('Input must contain at least two finite samples')
    start, end, listener, dims = map(np.array, [a.start, a.end, a.listener, a.room])
    if not all(np.all((v > 0) & (v < dims)) for v in [start, end, listener]):
        p.error('Source and listener must be strictly inside room')
    # Check the full line segment, not just its sampled endpoints.
    if closest_distance(start, end, listener) < 0.2:
        p.error('Trajectory passes too close to listener')
    audio, nodes, positions, rir_count = render_binaural(
        signal, bank, start, end, listener, a.mode, dims, a.absorption, a.order,
        a.step_ms, a.max_angle_step_deg, a.rir_step_ms, a.max_rir_distance_m,
        a.max_rir_radial_m, progress=True)
    # One gain for BOTH ears and the entire clip preserves ILD and distance changes.
    gain = min(1.0, 0.95/max(float(np.max(np.abs(audio))), 1e-15))
    audio *= gain
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.output, audio, bank.fs, subtype='PCM_24')
    relative = positions-listener
    metadata = {
        'implementation': 'simplified RIR-HRIR cascade; not official reproduction',
        'hrir_interpolation': 'local 48-neighbor onset-delay-aligned spherical Gaussian kernel, sigma=7.5deg',
        'distance_gain': '1/r on HRIR' if a.mode == 'hrtf' else 'from room RIR only',
        'step_ms': a.step_ms,
        'max_angle_step_deg': a.max_angle_step_deg,
        'hrtf_snapshot_count': int(len(nodes)),
        'rir_step_ms': a.rir_step_ms if a.mode == 'room' else None,
        'max_rir_distance_m': a.max_rir_distance_m if a.mode == 'room' else None,
        'max_rir_radial_m': a.max_rir_radial_m if a.mode == 'room' else None,
        'rir_snapshot_count': rir_count,
        'mode': a.mode, 'sample_rate': bank.fs, 'channels': ['left', 'right'],
        'sofa': str(a.sofa.absolute()), 'input': str(a.input) if a.input else 'seeded noise, seed=42',
        'room_m': a.room, 'listener_m': a.listener, 'absorption': a.absorption,
        'max_order': a.order, 'global_gain': gain,
        'coordinate_system': '+x front, +y left, +z up; positive azimuth left',
        'trajectory_time_basis': 'input/emission time; output includes propagation/filter delay and tail',
        'trajectory': [dict(time_s=float(t/bank.fs), xyz_m=pos.tolist(),
                            azimuth_deg=float(np.rad2deg(np.arctan2(v[1], v[0]))),
                            elevation_deg=float(np.rad2deg(np.arctan2(v[2], np.hypot(v[0], v[1])))),
                            distance_m=float(np.linalg.norm(v)))
                       for t, pos, v in zip(nodes, positions, relative)]}
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with a.output.with_suffix('.cues.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['output_time_s', 'itd_us_right_minus_left', 'ild_db_left_over_right'])
        writer.writerows(cue_rows(audio, bank.fs))
    print(f'Saved {a.output.absolute()} ({len(audio)/bank.fs:.2f}s, {bank.fs}Hz)')


if __name__ == '__main__':
    main()
