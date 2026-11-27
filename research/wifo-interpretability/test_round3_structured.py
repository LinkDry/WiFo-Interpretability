import os
from pathlib import Path
import unittest

import torch

from round3_structured import CutAttention, build_submodel, predict, probe_at
from wifo_pilot import TASKS

SOM = Path(os.environ.get('WIFO_SOM', 'E:/Projects/SoM'))
PLAN = Path(__file__).parent / 'results/round2_av_prune/prune_lock.json'


class StructuredTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and PLAN.is_file() and (SOM / 'WiFo/WiFo/wifo_base.pkl').is_file(),
                         'Local CUDA, WiFo checkpoint and Round 2 lock are required')
    @torch.inference_mode()
    def test_structured_matches_zero_gate_elementwise(self):
        reference = probe_at(SOM, 'cuda:0')
        sample = torch.randn((2, 2, 16, 32, 32), device='cuda',
                             generator=torch.Generator(device='cuda').manual_seed(817))
        for budget in (16, 23):
            with self.subTest(budget=budget):
                compact, removed = build_submodel(SOM, 'cuda:0', PLAN, budget)
                gates = {}
                for layer, head in removed:
                    gates.setdefault(layer, sample.new_ones((len(sample), 8)))[:, head] = 0
                self.assertEqual(sum(b.attn.num_heads for b in compact.blocks), 80 - budget)
                for layer in gates:
                    attn = compact.blocks[layer].attn
                    original = reference.blocks[layer].attn
                    self.assertIsInstance(attn, CutAttention)
                    self.assertEqual(attn.head_dim, 64)
                    self.assertEqual(attn.scale, original.scale)
                    self.assertEqual(attn.q.out_features, 64 * attn.num_heads)
                    torch.testing.assert_close(attn.proj.bias, original.proj.bias, rtol=0, atol=0)
                for task in TASKS:
                    baseline, _ = predict(reference, sample, task, gates)
                    result, _ = predict(compact, sample, task)
                    torch.testing.assert_close(result, baseline, atol=1e-5, rtol=3e-5)
                    print(f'Elementwise max abs ({budget}, {task}): {(result-baseline).abs().max().item():.3g}')


if __name__ == '__main__':
    unittest.main()
