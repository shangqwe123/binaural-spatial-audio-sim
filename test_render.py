import unittest
from unittest import mock
import numpy as np
from scipy.signal import fftconvolve
import render
from render import (render_snapshots, cue_rows, adaptive_hrtf_nodes,
                    adaptive_position_nodes, closest_distance, render_binaural)
from generate_dataset import sample_trajectory, motion_label


class RenderTests(unittest.TestCase):
    def test_static_matches_full_convolution_and_retains_tail(self):
        x = np.random.default_rng(1).normal(size=503)
        h = np.random.default_rng(2).normal(size=(2, 71))
        nodes = np.array([0, 100, 200, 300, 400, 502])
        y = render_snapshots(x, nodes, [h]*len(nodes))
        reference = np.stack([fftconvolve(x, ear) for ear in h], axis=1)
        np.testing.assert_allclose(y, reference, atol=1e-12)

    def test_itd_ild_sign_conventions(self):
        fs = 48000
        left = np.random.default_rng(3).normal(size=4800)
        right = np.r_[np.zeros(24), left[:-24]] * 0.5
        rows = np.array(list(cue_rows(np.c_[left, right], fs)))
        np.testing.assert_allclose(rows[:, 1], 500)
        self.assertTrue(np.all(np.abs(rows[:, 2]-6.0206) < 0.2))

    def test_time_varying_gain_has_no_block_steps(self):
        x = np.ones(101)
        filters = [np.array([[1.], [1.]]), np.array([[2.], [2.]])]
        y = render_snapshots(x, np.array([0, 100]), filters)
        np.testing.assert_allclose(y[:, 0], np.linspace(1, 2, 101), atol=1e-12)

    def test_static_source_uses_single_snapshot(self):
        p, listener = [4, 4, 1.5], [2, 2.5, 1.5]
        self.assertEqual(len(adaptive_hrtf_nodes(p, p, listener, 48000*5, 48000)[0]), 1)
        self.assertEqual(len(adaptive_position_nodes(p, p, 48000*5, 48000)[0]), 1)

    def test_radial_bound_adds_snapshots_only_for_radial_motion(self):
        fs, n, listener = 48000, 48000, np.zeros(3)
        args = dict(max_distance_m=10, max_step_ms=10000, listener=listener, max_radial_m=0.02)
        radial = adaptive_position_nodes([1, 0, 0], [2, 0, 0], n, fs, **args)[0]
        self.assertGreaterEqual(len(radial)-1, 50)
        # 0.69 m tangential chord: radial change 2*(r - r*cos(10deg)) ~= 0.06 m.
        a = np.deg2rad(10)
        chord = adaptive_position_nodes([2*np.cos(a), 2*np.sin(a), 0],
                                        [2*np.cos(a), -2*np.sin(a), 0], n, fs, **args)[0]
        self.assertLessEqual(len(chord)-1, 4)

    def test_room_mode_applies_distance_attenuation_once(self):
        class FlatBank:
            fs = 1000
            def at(self, direction):
                return np.ones((2, 1))
        def delta_rir(source, listener, *args):
            return np.array([1/np.linalg.norm(np.asarray(source)-listener)])
        listener, source = np.array([1., 1., 1.]), np.array([3., 1., 1.])
        with mock.patch.object(render, 'room_rir', delta_rir):
            y = render_binaural(np.ones(10), FlatBank(), source, source, listener,
                                'room', room_dims=[5, 5, 3])[0]
        np.testing.assert_allclose(y[:, 0], 0.5)


class DatasetTests(unittest.TestCase):
    def test_trajectories_cover_all_sides_and_respect_limits(self):
        dims = np.array([6., 5., 3.])
        listener = dims*np.array([.35, .5, .5])
        static_az, moving_az = [], []
        for i in range(400):
            rng = np.random.default_rng([1, i])
            static = i % 4 == 3
            start, end = sample_trajectory(rng, listener, dims, static, min_distance=1.0)
            self.assertGreaterEqual(closest_distance(start, end, listener), 1.0)
            for v in (start, end):
                self.assertTrue(np.all((v > 0) & (v < dims)))
            v = start-listener
            (static_az if static else moving_az).append(np.rad2deg(np.arctan2(v[1], v[0])))
        static_az, moving_az = np.array(static_az), np.array(moving_az)
        self.assertTrue((static_az > 0).any() and (static_az < 0).any())
        self.assertTrue((np.abs(moving_az) > 120).any())  # rear hemisphere

    def test_motion_label(self):
        listener = np.zeros(3)
        self.assertEqual(motion_label(np.array([1., 1, 0]), np.array([1., -1, 0]), listener),
                         'azimuth_decreasing')
        # Rear crossing through 180 deg must not be confused by wrap-around.
        self.assertEqual(motion_label(np.array([-1., 1, 0]), np.array([-1., -1, 0]), listener),
                         'azimuth_increasing')
        p = np.array([1., 0, 0])
        self.assertEqual(motion_label(p, p, listener), 'static')


if __name__ == '__main__':
    unittest.main()
