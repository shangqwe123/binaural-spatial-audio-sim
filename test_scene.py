import unittest
from pathlib import Path
import numpy as np
from captions import make_captions
from prepare_sources import MULTI_SOURCE
from scene import (C, DISTANCES, ROOMS, SECTORS, adaptive_nodes, distance_class, image_sources,
                   late_tail, place_listener, positions, sample_motion, sample_room, sector_of)

SOFA = Path(__file__).parent/'data/hrtf_nh2.sofa'


class SceneTests(unittest.TestCase):
    def test_sector_and_distance_labels(self):
        self.assertEqual(sector_of(0), 'front')
        self.assertEqual(sector_of(90), 'left')
        self.assertEqual(sector_of(-135), 'back_right')
        self.assertEqual(sector_of(179), 'back')
        self.assertEqual(sector_of(-179), 'back')
        self.assertEqual(distance_class(1.2), 'near')
        self.assertEqual(distance_class(4.0), 'far')

    def test_motions_change_one_attribute_and_fit_rooms(self):
        seen = set()
        for i in range(600):
            rng = np.random.default_rng([3, i])
            kind = list(ROOMS)[i % len(ROOMS)]
            room = sample_room(rng, kind)
            traj = sample_motion(rng, kind)
            seen.add((kind, traj['kind']))
            start, end = positions(traj, np.zeros(3), [0, 1])
            r0, r1 = np.linalg.norm(start), np.linalg.norm(end)
            if traj['kind'] in ('static', 'arc', 'line'):
                self.assertAlmostEqual(r0, r1)
                self.assertIn(distance_class(r0), ROOMS[kind]['distances'])
            if traj['kind'] == 'radial':
                self.assertEqual(sector_of(traj['az0']), sector_of(traj['az1']))
                self.assertEqual({distance_class(r0), distance_class(r1)}, {'near', 'far'})
            if traj['kind'] == 'arc':
                path = np.linalg.norm(positions(traj, np.zeros(3), np.linspace(0, 1, 50)), axis=1)
                np.testing.assert_allclose(path, r0)
                # 2-4 sectors apart; each end jitters +/-15 deg around its centre.
                self.assertTrue(60-1e-9 <= abs(traj['az1']-traj['az0']) <= 210+1e-9)
            if traj['kind'] == 'line':
                path = np.linalg.norm(positions(traj, np.zeros(3), np.linspace(0, 1, 50)), axis=1)
                self.assertGreaterEqual(path.min(), 0.75*DISTANCES[distance_class(r0)][0]-1e-6)
            # Endpoints stay well inside their sector.
            for az in (traj['az0'], traj['az1']):
                centre = 45*SECTORS.index(sector_of(az))
                self.assertLessEqual(abs((az-centre+180) % 360-180), 15+1e-9)
            listener = place_listener(rng, room, traj)
            if listener is not None and room['dims'] is not None:
                pts = positions(traj, listener, np.linspace(0, 1, 64))
                self.assertTrue(np.all((pts > 0) & (pts < room['dims'])))
        self.assertIn(('room', 'radial'), seen)
        self.assertIn(('hall', 'radial'), seen)

    def test_image_sources(self):
        room = {'dims': np.array([4., 3., 2.5]), 'beta': 0.8}
        points, gains = image_sources([1., 1., 1.], room, max_order=1)
        self.assertEqual(len(points), 7)  # direct + one per wall
        np.testing.assert_allclose(points[0], [1, 1, 1])
        self.assertIn([-1., 1., 1.], points.tolist())  # mirror in x=0
        self.assertIn([7., 1., 1.], points.tolist())   # mirror in x=4
        np.testing.assert_allclose(sorted(gains), [.8]*6+[1])
        self.assertEqual(len(image_sources([1., 1., 1.], room, max_order=2)[0]), 25)
        open_points, open_gains = image_sources([1., 2., 1.5], {'dims': None})
        np.testing.assert_allclose(open_points[1], [1, 2, -1.5])

    def test_adaptive_nodes_bound_angle_and_distance(self):
        listener, fs, n = np.zeros(3), 48000, 48000*4
        static = {'kind': 'static', 'el': 0, 'az0': 30, 'az1': 30, 'r0': 2, 'r1': 2}
        pos = lambda s: positions(static, listener, np.asarray(s)/(n-1))
        self.assertEqual(len(adaptive_nodes(pos, n, fs, listener)), 1)
        radial = {'kind': 'radial', 'el': 0, 'az0': 30, 'az1': 30, 'r0': 4, 'r1': 1}
        pos = lambda s: positions(radial, listener, np.asarray(s)/(n-1))
        nodes = adaptive_nodes(pos, n, fs, listener, max_step_ms=1e6)
        steps = np.abs(np.diff(np.linalg.norm(pos(nodes), axis=1)))
        self.assertLessEqual(steps.max(), 0.02+1e-9)

    def test_tail_energy_matches_critical_distance(self):
        room = {'dims': np.array([8., 6., 3.]), 'rt60': 0.6}
        tail = late_tail(np.random.default_rng(0), room, 48000, 0.005)
        r_c = 0.057*np.sqrt(144/0.6)
        np.testing.assert_allclose(np.mean(np.sum(tail**2, axis=1)), 1/r_c**2, rtol=1e-6)
        # Independent ears: low interaural correlation.
        self.assertLess(abs(np.corrcoef(tail)[0, 1]), 0.1)

    @unittest.skipUnless(SOFA.is_file(), 'needs data/hrtf_nh2.sofa')
    def test_direct_path_delay_and_side(self):
        from render import HrirBank
        from scene import Hrirs, early_brir, early_length
        hrirs = Hrirs(HrirBank(SOFA))
        room = {'dims': None}
        listener, source = np.array([0., 0, 1.5]), np.array([0., 2., 1.5])  # 2 m to the left
        h = early_brir(hrirs, source, listener, room, early_length([source], listener, room, hrirs.fs, 512))
        self.assertGreater(np.sum(h[0]**2), 4*np.sum(h[1]**2))  # left ear louder
        onset = np.argmax(np.abs(h[0]) > 0.1*np.abs(h[0]).max())
        self.assertAlmostEqual(onset/hrirs.fs, 2/C, delta=0.002)


class PlanTests(unittest.TestCase):
    def test_plan_follows_ratios(self):
        from collections import Counter
        from generate_dataset import DEFAULT_RATIOS, plan_variants
        counts = {k: Counter() for k in ('room', 'motion', 'distance', 'sector')}
        for seed in range(3000):
            plans = plan_variants(np.random.default_rng(seed), 8)
            self.assertEqual({p['sector'] for p in plans}, set(SECTORS))  # 1/8 each, n=8
            for p in plans:
                counts['room'][p['room']] += 1
                counts['sector'][p['sector']] += 1
                counts['motion'][p['motion'] if p['motion'] != 'radial'
                                 else ('approach' if p['approach'] else 'recede')] += 1
                if p['distance']:
                    counts['distance'][p['distance']] += 1
        for group, key in (('scene', 'room'), ('motion', 'motion'), ('distance', 'distance')):
            total = sum(counts[key].values())
            for value, share in DEFAULT_RATIOS[group].items():
                self.assertAlmostEqual(counts[key][value]/total, share, delta=0.01)

    def test_custom_ratios_drop_zero_shares(self):
        from generate_dataset import DEFAULT_RATIOS, plan_variants
        ratios = dict(DEFAULT_RATIOS, scene={'hall': 1.0}, motion={'static': 1.0})
        plans = plan_variants(np.random.default_rng(0), 5, ratios)
        self.assertEqual({p['room'] for p in plans}, {'hall'})
        self.assertEqual({p['motion'] for p in plans}, {'static'})


class CaptionTests(unittest.TestCase):
    def test_captions(self):
        spatial = {'motion': 'arc', 'start': {'sector': 'front_left', 'distance_class': 'near'},
                   'end': {'sector': 'back_right', 'distance_class': 'near'}, 'via_sector': 'front_right'}
        caps = make_captions({'type': 'sfx', 'text': 'A dog barks.'}, 'room', spatial)
        self.assertEqual(caps['zh'], '在一个房间里，声音：A dog barks，声音在近处从左前方经右前方绕到右后方。')
        self.assertIn('a dog barks, circling nearby from the front left via the front right', caps['en'])

    def test_multi_source_filter(self):
        self.assertIsNone(MULTI_SOURCE.search('A dog barks repeatedly'))
        self.assertIsNotNone(MULTI_SOURCE.search('A man speaks while birds chirp'))


if __name__ == '__main__':
    unittest.main()
