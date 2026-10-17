"""Binaural scene simulation used by generate_dataset.py.

Coordinates as in render.py: +x front, +y left, +z up; azimuth positive to the
left. Distances in metres; amplitudes are referenced to 1 m (direct gain 1/r).

Early part: direct path plus shoebox image sources up to EARLY_ORDER, each
filtered by the HRIR of its own arrival direction (render.py's room mode sends
every reflection through the direct-path HRIR instead). Late part: decaying
noise, independent per ear (diffuse field), with energy from Sabine's critical
distance and a three-band decay. Simplifications: frequency-independent wall
absorption for early reflections, fixed head, no Doppler, no near-field HRTF.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from render import render_snapshots

C = 343.0
EARLY_ORDER = 2
GROUND_REFLECTION = 0.7  # outdoor ground, pressure reflection coefficient

# Sector k is centred on azimuth 45*k degrees (positive = left).
SECTORS = ['front', 'front_left', 'left', 'back_left',
           'back', 'back_right', 'right', 'front_right']
SECTOR_JITTER_DEG = 15  # stay well inside a 45-degree sector
DISTANCES = {'near': (1.0, 1.5), 'far': (3.0, 5.0)}
ROOMS = {
    # dims ranges (x, y, z) in metres. 'room' spans bedroom to living room or
    # classroom; far sources only fit the larger ones (placement retries).
    'room': {'dims': [(3, 10), (3, 8), (2.5, 3.5)], 'rt60': (0.2, 0.6), 'distances': ['near', 'far']},
    'hall': {'dims': [(15, 30), (12, 20), (6, 10)], 'rt60': (0.9, 1.6), 'distances': ['near', 'far']},
    'open': {'dims': None, 'rt60': None, 'distances': ['near', 'far']},
}
MOTIONS = {'static': 0.4, 'arc': 0.15, 'line': 0.15, 'radial': 0.3}  # used when no plan fixes the motion
WALL_MARGIN = 0.3
LINE_MIN_FRACTION = 0.75  # closest point of a line vs. its distance band's lower edge
RADIAL_NEAR, RADIAL_FAR = (1.0, 1.25), (4.0, 5.0)
MOVE_SECONDS = (2.0, 4.0)  # roughly walking pace; the source holds still before/after


def wrap_deg(a):
    return (np.asarray(a, dtype=float)+180) % 360-180


def sector_of(azimuth):
    return SECTORS[int(np.round(wrap_deg(azimuth)/45)) % 8]


def distance_class(r):
    for name, (lo, hi) in DISTANCES.items():
        if lo-1e-9 <= r <= hi+1e-9:
            return name
    return 'mid'


def sample_room(rng, kind):
    spec = ROOMS[kind]
    if spec['dims'] is None:
        return {'kind': kind, 'dims': None, 'rt60': None, 'beta': None}
    dims = np.array([rng.uniform(*r) for r in spec['dims']])
    rt60 = float(rng.uniform(*spec['rt60']))
    volume = float(np.prod(dims))
    surface = 2*(dims[0]*dims[1]+dims[0]*dims[2]+dims[1]*dims[2])
    alpha = float(np.clip(0.161*volume/(surface*rt60), 0.01, 0.99))  # Sabine
    return {'kind': kind, 'dims': dims, 'rt60': rt60, 'alpha': alpha, 'beta': np.sqrt(1-alpha)}


def sample_motion(rng, room_kind, kind=None, sector=None, dist=None, approach=None):
    """Draw one trajectory; every motion changes a single attribute.

    static: nothing; arc: azimuth at fixed distance (90-180 deg); line: straight
    90-degree chord (passes ~0.7x closer mid-way); radial: distance at fixed
    azimuth (far<->near). kind/sector (start sector)/dist/approach fix those
    attributes; anything left as None is drawn at random.
    """
    distances = ROOMS[room_kind]['distances']
    if kind is None:
        kinds = [k for k in MOTIONS if k != 'radial' or 'far' in distances]
        p = np.array([MOTIONS[k] for k in kinds])
        kind = str(rng.choice(kinds, p=p/p.sum()))
    k0 = SECTORS.index(sector) if sector is not None else int(rng.integers(8))
    az0 = 45*k0 + rng.uniform(-SECTOR_JITTER_DEG, SECTOR_JITTER_DEG)
    traj = {'kind': kind, 'el': float(rng.uniform(-10, 10)), 'az0': float(wrap_deg(az0))}
    if kind == 'radial':
        # Wider than the static bands (ratio 3.2-5x, ~10-14 dB of direct level).
        near, far = rng.uniform(*RADIAL_NEAR), rng.uniform(*RADIAL_FAR)
        if approach is None:
            approach = bool(rng.random() < 0.5)
        traj.update(az1=traj['az0'], r0=far if approach else near, r1=near if approach else far)
        return traj
    if dist is None:
        dist = str(rng.choice(distances))
    lo, hi = DISTANCES[dist]
    if kind == 'static':
        r = rng.uniform(lo, hi)
        traj.update(az1=traj['az0'], r0=r, r1=r)
        return traj
    steps = int(rng.integers(2, 5)) if kind == 'arc' else 2
    sign = int(rng.choice([-1, 1]))
    while True:
        jitter = rng.uniform(-SECTOR_JITTER_DEG, SECTOR_JITTER_DEG)
        # az1 stays unwrapped so an arc knows which way round it goes.
        az1 = traj['az0'] + (45*k0 + sign*45*steps + jitter - az0)
        # A chord dips to r*cos(sweep/2) mid-way: start from the outer part of
        # the band and keep the closest point >= 75% of the band's lower edge.
        r = rng.uniform((lo+hi)/2, hi) if kind == 'line' else rng.uniform(lo, hi)
        if kind == 'arc' or r*np.cos(np.deg2rad(az1-traj['az0'])/2) >= LINE_MIN_FRACTION*lo:
            break
    traj.update(az1=float(az1), r0=r, r1=r)
    return traj


def sample_timing(rng, traj, active, fs):
    """Hold - move - hold: a 2-4 s move placed inside the active span.

    Spreading the move over a whole (possibly 15 s) clip makes it too slow to
    hear against the natural level changes of speech. active = (first, last)
    sample of the non-silent part. Sets traj['move'] = [start, end] samples.
    """
    if traj['kind'] == 'static':
        return traj
    first, last = active
    span = last-first
    length = min(span, int(rng.uniform(*MOVE_SECONDS)*fs))
    start = first + int(rng.integers(0, span-length+1))
    traj['move'] = [start, start+max(length, 1)]
    return traj


def path_fraction(traj, samples, n):
    """Fraction of the path covered at each sample (linear within the move)."""
    samples = np.asarray(samples, dtype=float)
    start, end = traj.get('move', (0, max(n-1, 1)))
    return np.clip((samples-start)/max(end-start, 1), 0, 1)


def positions(traj, listener, u):
    """Source xyz at path fractions u in [0, 1]."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    el = np.deg2rad(traj['el'])

    def point(az, r):
        az = np.deg2rad(az)
        return np.asarray(listener) + np.stack(
            [r*np.cos(el)*np.cos(az), r*np.cos(el)*np.sin(az), r*np.sin(el)*np.ones_like(az)], -1)
    if traj['kind'] == 'arc':
        return point(traj['az0']+(traj['az1']-traj['az0'])*u, traj['r0']*np.ones_like(u))
    a = point(np.array([traj['az0']]), np.array([traj['r0']]))
    b = point(np.array([traj['az1']]), np.array([traj['r1']]))
    return a + (b-a)*u[:, None]


def place_listener(rng, room, traj, tries=200):
    """Random listener position such that the whole trajectory stays inside."""
    if room['dims'] is None:
        listener = np.array([0., 0., rng.uniform(1.2, 1.7)])
        return listener if np.all(positions(traj, listener, np.linspace(0, 1, 64))[:, 2] > 0.1) else None
    dims = room['dims']
    for _ in range(tries):
        listener = np.array([rng.uniform(0.5, dims[0]-0.5), rng.uniform(0.5, dims[1]-0.5),
                             rng.uniform(1.2, min(1.7, dims[2]-0.5))])
        pts = positions(traj, listener, np.linspace(0, 1, 64))
        if np.all((pts > WALL_MARGIN) & (pts < dims-WALL_MARGIN)):
            return listener
    return None


def adaptive_nodes(position, n, fs, listener, max_angle_deg=1.0, max_step_ms=20.0, max_radial_m=0.02):
    """Snapshot samples bounding time, direction change and distance change.

    The distance bound keeps crossfaded snapshots' direct-path delays within a
    few samples of each other (no audible comb filtering).
    """
    ends = position(np.array([0, max(n-1, 0)]))
    if n < 2 or np.allclose(ends[0], ends[1]):
        return np.array([0])
    max_angle = np.deg2rad(max_angle_deg)
    max_samples = max(1, round(fs*max_step_ms/1000))
    nodes, left = [0], 0
    while left < n-1:
        right = min(n-1, left+max_samples)
        while right-left > 1:
            mid = (left+right)//2
            v = position(np.array([left, mid, right]))-listener
            r = np.linalg.norm(v, axis=1)
            d = v/r[:, None]
            angle = max(np.arccos(np.clip(d[0] @ d[1], -1, 1)), np.arccos(np.clip(d[1] @ d[2], -1, 1)),
                        np.arccos(np.clip(d[0] @ d[2], -1, 1)))
            radial = max(abs(r[0]-r[1]), abs(r[1]-r[2]), abs(r[0]-r[2]))
            if angle <= max_angle and radial <= max_radial_m:
                break
            right = mid
        nodes.append(right)
        left = right
    return np.asarray(nodes)


class Hrirs:
    """HRIR lookup scaled to unit diffuse-field energy; reflections use a 2-degree cache.

    Scale and diffuse-field response come from the interpolated HRIRs actually
    used (interpolation smooths away part of the measured energy), so direct
    sound and the diffuse tail share one reference.
    """
    def __init__(self, bank, quant_deg=2.0):
        self.bank, self.fs, self.quant = bank, bank.fs, quant_deg
        sample =[bank.at(d) for d in bank.directions[::max(1, len(bank.directions)//300)]]
        power = np.mean([np.abs(np.fft.rfft(h, axis=-1))**2 for h in sample], axis=(0, 1))
        self.scale = 1/np.sqrt(np.mean([np.sum(h**2, axis=-1) for h in sample]))
        # Zero-phase FIR with the diffuse-field magnitude (unit energy) for the tail.
        taps = sample[0].shape[-1]
        fir = np.roll(np.fft.irfft(np.sqrt(power), n=taps), taps//2)*np.hanning(taps)
        self.diffuse_fir = fir/np.sqrt(np.sum(fir**2))
        self.cache = {}

    def exact(self, direction):
        return self.bank.at(direction)*self.scale

    def quantised(self, direction):
        az = np.rad2deg(np.arctan2(direction[1], direction[0]))
        el = np.rad2deg(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))
        key = (int(round(az/self.quant)), int(round(el/self.quant)))
        if key not in self.cache:
            a, e = np.deg2rad(key[0]*self.quant), np.deg2rad(key[1]*self.quant)
            self.cache[key] = self.exact(np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]))
        return self.cache[key]


def image_sources(source, room, max_order=EARLY_ORDER):
    """Shoebox image sources (Allen & Berkley); open space keeps only the ground."""
    source = np.asarray(source, dtype=float)
    if room['dims'] is None:
        ground = source*np.array([1, 1, -1])
        return np.stack([source, ground]), np.array([1.0, GROUND_REFLECTION])
    axes = []
    for s, length in zip(source, room['dims']):
        # Image coordinate (1-2u)*s + 2*n*L hits |n-u| + |n| walls.
        axes.append([((1-2*u)*s+2*n*length, abs(n-u)+abs(n))
                     for n in range(-max_order, max_order+1) for u in (0, 1)
                     if abs(n-u)+abs(n) <= max_order])
    points, orders = [], []
    for x, ox in axes[0]:
        for y, oy in axes[1]:
            for z, oz in axes[2]:
                if ox+oy+oz <= max_order:
                    points.append((x, y, z))
                    orders.append(ox+oy+oz)
    orders = np.asarray(orders)
    first = np.argmin(orders)  # direct path first
    order = np.r_[first, np.delete(np.arange(len(orders)), first)]
    return np.asarray(points)[order], room['beta']**orders[order]


def early_length(points, listener, room, fs, taps):
    """Common filter length for early BRIRs at all given source points."""
    longest = max(np.linalg.norm(image_sources(p, room)[0]-listener, axis=1).max() for p in points)
    return int(np.ceil(longest/C*fs))+taps+2


def early_brir(hrirs, source, listener, room, length):
    """Direct + early reflections, each through its own direction's HRIR.

    The direct path gets an exact fractional delay (it dominates localisation
    and snapshot crossfades); reflections use two-tap linear interpolation.
    """
    points, gains = image_sources(source, room)
    vectors = points-listener
    dist = np.linalg.norm(vectors, axis=1)
    delays = dist/C*hrirs.fs
    out = np.zeros((2, length))
    direct = hrirs.exact(vectors[0])
    taps = direct.shape[-1]
    whole, frac = int(delays[0]), delays[0]-int(delays[0])
    nfft = 2*taps
    shifted = np.fft.irfft(np.fft.rfft(direct, n=nfft, axis=-1) *
                           np.exp(-2j*np.pi*frac*np.fft.rfftfreq(nfft)), n=nfft, axis=-1)
    out[:, whole:whole+nfft] += shifted[:, :length-whole]*gains[0]/dist[0]
    for v, delay, g, d in zip(vectors[1:], delays[1:], gains[1:], dist[1:]):
        h = hrirs.quantised(v)*(g/d)
        whole, frac = int(delay), delay-int(delay)
        out[:, whole:whole+taps] += (1-frac)*h
        out[:, whole+1:whole+1+taps] += frac*h
    return out


def late_tail(rng, room, fs, onset_s, diffuse_fir=None, early_energy=0.0):
    """Diffuse late reverberation: independent noise per ear, three-band decay.

    diffuse_fir, if given, colours the noise like a diffuse-field HRTF.

    All reverberant energy per ear (early reflections + this tail) equals the
    direct-path energy at Sabine's critical distance r_c = 0.057*sqrt(V/RT60),
    i.e. direct = reverberant at r_c. early_energy is what the explicit image
    sources already contribute; the tail supplies the rest (at least 20%).
    """
    dims, rt60 = room['dims'], room['rt60']
    r_c = 0.057*np.sqrt(np.prod(dims)/rt60)
    mixing = 2*np.mean(dims)/C  # fade-in while early reflections dominate
    bands = [('lowpass', 500, 1.15), ('bandpass', (500, 4000), 1.0), ('highpass', 4000, 0.6)]
    length = int((onset_s + 1.2*rt60*max(b[2] for b in bands))*fs)
    t = np.arange(length)/fs - onset_s
    fade = 0.5-0.5*np.cos(np.pi*np.clip(t/mixing, 0, 1))
    tail = np.zeros((2, length))
    for kind, edge, scale in bands:
        # Flat-magnitude, random-phase noise: the ears are uncorrelated but get
        # identical spectra, so the diffuse tail adds no random ILD.
        phases = np.exp(2j*np.pi*rng.random((2, length//2+1)))
        noise = np.fft.irfft(phases, n=length, axis=-1)*np.sqrt(length)
        noise = sosfilt(butter(4, edge, kind, fs=fs, output='sos'), noise, axis=-1)
        tail += noise*np.exp(-6.908/(rt60*scale)*np.clip(t, 0, None))
    if diffuse_fir is not None:
        tail = np.stack([np.convolve(ear, diffuse_fir, mode='same') for ear in tail])
    tail *= fade
    target = max((1/r_c)**2-early_energy, 0.2*(1/r_c)**2)
    tail *= np.sqrt(target/np.mean(np.sum(tail**2, axis=1)))
    return tail


def reflection_energy(source, listener, room):
    """Per-ear energy of the explicit reflections (HRIRs have unit mean energy)."""
    points, gains = image_sources(source, room)
    dist = np.linalg.norm(points[1:]-listener, axis=1)
    return float(np.sum((gains[1:]/dist)**2))


def render_scene(signal, hrirs, traj, listener, room, rng,
                 max_angle_deg=1.0, max_step_ms=20.0, max_radial_m=0.02):
    """Return (binaural audio [samples, 2], snapshot nodes, snapshot positions)."""
    n = len(signal)
    position = lambda samples: positions(traj, listener, path_fraction(traj, samples, n))
    nodes = adaptive_nodes(position, n, hrirs.fs, listener, max_angle_deg, max_step_ms, max_radial_m)
    points = position(nodes)
    length = early_length(points, listener, room, hrirs.fs, 2*hrirs.bank.ir.shape[-1])
    audio = render_snapshots(signal, nodes, [early_brir(hrirs, p, listener, room, length)
                                             for p in points])
    if room['rt60']:
        onset = np.min(np.linalg.norm(positions(traj, listener, np.linspace(0, 1, 64))-listener, axis=1))/C
        early = np.mean([reflection_energy(p, listener, room)
                         for p in positions(traj, listener, np.linspace(0, 1, 5))])
        tail = late_tail(rng, room, hrirs.fs, onset, hrirs.diffuse_fir, early)
        late = np.stack([fftconvolve(signal, ear) for ear in tail], axis=1)
        total = np.zeros((max(len(audio), len(late)), 2))
        total[:len(audio)] += audio
        total[:len(late)] += late
        audio = total
    return audio, nodes, points
