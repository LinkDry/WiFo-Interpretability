import unittest
from pathlib import Path

import numpy as np
import torch

from round2_experiment import av_site, build_blind_groups, lock_count, make_probe, tv_distance
from wifo_pilot import Grid, HEADS, Probe, TASKS, prepared_input


class Round2Tests(unittest.TestCase):
    def test_budget_uses_worst_domain_and_checks_nonmonotone_prefixes(self):
        effects = np.array([[0, .19, .24, .18, .27], [0, .10, .10, .20, .18]])
        self.assertEqual(lock_count(effects, .2), 3)

    def test_layer_random_preserves_budget(self):
        ranking = [list(HEADS), list(reversed(HEADS))]
        groups = build_blind_groups(ranking, list(HEADS), [[13] * 3, [21] * 3])
        for task in TASKS:
            entry = groups[task]['0.2']
            counts = np.bincount([x[0] for x in entry['Task-Aware'][0]], minlength=10)
            self.assertEqual(len(entry['Same-Layer Random']), 10)
            for sample in entry['Same-Layer Random']:
                self.assertEqual(len(set(sample)), len(sample))
                np.testing.assert_array_equal(counts, np.bincount([x[0] for x in sample], minlength=10))

    def test_joint_ablation_masks_variant_rows_and_leaves_baseline_active(self):
        records = []

        class Spy:
            def error(self, data, target, task, gates):
                records.append({layer: value.clone() for layer, value in gates.items()})
                return data[:, 0, 0, 0, 0]

        data = torch.tensor([1., 2.]).reshape(2, 1, 1, 1, 1)
        results = Probe.variant_errors(Spy(), data, data.reshape(2, 1, 1), 'temporal',
                                       [(), ((0, 1), (0, 3), (6, 2)), ((6, 5),)], 2)
        np.testing.assert_array_equal(results, [[1., 2.]] * 3)
        self.assertEqual(set(records[0]), {0, 6})
        layer, observation = torch.where(records[0][0] == 0)
        self.assertEqual(list(zip(layer.tolist(), observation.tolist())), [(2, 1), (2, 3), (3, 1), (3, 3)])
        rows, cols = torch.where(records[0][6] == 0)
        self.assertEqual(list(zip(rows.tolist(), cols.tolist())), [(2, 2), (3, 2)])
        rows, cols = torch.where(records[1][6] == 0)
        self.assertEqual(list(zip(rows.tolist(), cols.tolist())), [(0, 5), (1, 5)])

    def test_total_variation(self):
        a = torch.tensor([[[[.5, .5], [.25, .75]]]])
        b = torch.tensor([[[[1., 0.], [.25, .75]]]])
        mask = torch.tensor([True, True])
        torch.testing.assert_close(tv_distance(a, a, mask), torch.tensor([[0.]]))
        torch.testing.assert_close(tv_distance(a, b, mask), torch.tensor([[.25]]))

    @unittest.skipUnless(torch.cuda.is_available() and Path('E:/Projects/SoM/WiFo/WiFo/wifo_base.pkl').is_file(),
                         'Requires the local frozen WiFo checkpoint and CUDA')
    @torch.inference_mode()
    def test_identity_replacement_matches_original_both_tasks_all_five_heads(self):
        probe = make_probe(Path('E:/Projects/SoM'), 'cuda:0')
        gen = torch.Generator(device=probe.device).manual_seed(86)
        clean = torch.randn((2, 2, 16, 32, 32), device=probe.device, generator=gen)
        target = probe.target(clean)
        for task in TASKS:
            data = prepared_input(clean, task, gen)
            predictions = []
            handle = probe.model.decoder_pred.register_forward_hook(
                lambda module, inputs, output: predictions.append(output.detach().clone()))
            self.addCleanup(handle.remove)
            baseline = probe.error(data, target, task)
            baseline_prediction = predictions[-1]
            captures = []
            with av_site(probe, capture=captures):
                probe.error(data, target, task)
            for head in (0, 1, 2, 4, 7):
                attn, values = (tensor[:, head] for tensor in captures[0])
                with av_site(probe, head=head, donor_a=attn, donor_v=values):
                    identity = probe.error(data, target, task)
                torch.testing.assert_close(identity, baseline, rtol=2e-5, atol=1e-6)
                torch.testing.assert_close(predictions[-1], baseline_prediction, rtol=2e-5, atol=1e-6)
            handle.remove()


if __name__ == '__main__':
    unittest.main()
