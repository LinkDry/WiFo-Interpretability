import unittest

import numpy as np
import torch

from wifo_pilot import Grid, TappedChannel, comparison_ci, gate_input


class PilotChecks(unittest.TestCase):
    def test_projected_channels_are_gated_only_for_selected_examples_and_heads(self):
        channels = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4)
        gates = torch.tensor([[0., 1.], [1., 1.]])
        actual = gate_input(channels, gates)
        expected = channels.clone()
        expected[0, :, :2] = 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prediction_regions_have_same_count_on_pilot_grid(self):
        grid = Grid((16, 32, 32))
        for task in ('temporal', 'fre'):
            mask = grid.visible(task)
            self.assertEqual(mask.shape, (256,))
            self.assertEqual(mask.sum().item(), 128)
        # Row-major order, frequency is the innermost coordinate.
        self.assertEqual(grid.visible('fre')[:8].tolist(), [True] * 4 + [False] * 4)
        self.assertFalse(grid.visible('temporal')[128])

    def test_factors_are_exact_and_paired(self):
        paths = TappedChannel.draw(5, seed=42, profile='diffuse')
        baseline, _, _, both = paths.render()
        np.testing.assert_allclose(paths.weighted_spread(paths.doppler), 1)
        np.testing.assert_allclose(paths.weighted_spread(paths.delays), 1)
        np.testing.assert_allclose(baseline[:, :, 0, :, 0], both[:, :, 0, :, 0], atol=1e-6)
        self.assertFalse(np.allclose(baseline, both))

    def test_bootstrap_keeps_paired_samples(self):
        values = np.array([[.5, 1., 2., 4.], [1., 2., 4., 8.]])
        interval = comparison_ci([(values, [-1, 1])], repeats=200)
        np.testing.assert_allclose(interval, 10 * np.log10(2))


if __name__ == '__main__':
    unittest.main()
