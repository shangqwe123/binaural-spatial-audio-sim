import unittest
import numpy as np
from scipy.signal import fftconvolve
from render import render_snapshots, cue_rows


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


if __name__ == '__main__':
    unittest.main()
