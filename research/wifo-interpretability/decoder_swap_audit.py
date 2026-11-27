"""Exhaustive decoder layer-1 map swap on the original paired synthetic panels."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wifo_pilot import Grid, Probe, TASKS, TappedChannel, as_tensor, atomic_npz, comparison_ci, prepared_input, write_table


def swaps_for_profile(probe, source, batch):
    paths = TappedChannel(*(source[key] for key in ('path_gains', 'steering', 'doppler_relative', 'delay_relative', 'path_power')))
    csi = paths.render()
    grid = Grid((16, 32, 32))
    layer = 6
    output = []
    for task in TASKS:
        task_batches = []
        for start in range(0, len(csi[0]), batch):
            clean = [as_tensor(condition[start:start + batch], probe) for condition in csi]
            recipient_inputs = [prepared_input(condition, task, torch.Generator(device=probe.device).manual_seed(8515 + start))
                                for condition in clean]
            targets = [probe.target(condition) for condition in clean]
            head_order = [(layer, head) for head in range(8)]
            captures = []
            baseline = []
            for item, target in zip(recipient_inputs, targets):
                collector = {}
                with probe.attention(head_order, task, grid, capture=collector):
                    errors = probe.error(item, target, task)
                captures.append(collector[layer])
                baseline.append(errors)
            results = []
            for factor in (0, 1, 2):
                cell_results = []
                for cell in range(4):
                    item, target, original = recipient_inputs[cell], targets[cell], captures[cell]
                    use_donor = captures[cell ^ factor]
                    variants = original.unsqueeze(0).repeat(8, 1, 1, 1, 1)
                    for head in range(8):
                        variants[head, :, head] = use_donor[:, head]
                    with probe.attention(head_order, task, grid, donor={layer: variants.flatten(0, 1)}):
                        replacement = probe.error(item.repeat(8, 1, 1, 1, 1), target.repeat(8, 1, 1), task).reshape(8, len(item))
                    if factor == 0:
                        torch.testing.assert_close(replacement, baseline[cell].expand_as(replacement), rtol=3e-5, atol=1e-6)
                    cell_results.append(torch.stack((baseline[cell].expand_as(replacement), replacement)).cpu().numpy())
                results.append(np.stack(cell_results))
            task_batches.append(np.stack(results[1:]))
        output.append(np.concatenate(task_batches, axis=-1))
    return np.stack(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--som', type=Path, default=Path('E:/Projects/SoM'))
    args = parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    probe = Probe(args.som, 'cuda:0')
    run_config = json.loads((args.output / 'run.json').read_text(encoding='utf-8'))
    batches = run_config['physics_batch']
    rows = []
    for profile in run_config['physics_seeds']:
        with np.load(args.output / f'physics_{profile}.npz') as source:
            result = swaps_for_profile(probe, source, batches)
            np.testing.assert_allclose(result[:, 0, :, 0, 0], source['swaps'][:, 0, 0, :, 0], rtol=3e-5, atol=1e-6)
        atomic_npz(args.output / f'decoder_l1_{profile}.npz', errors=result)
        for task, name in enumerate(TASKS):
            weights = np.repeat([1 if task == k else -1 for k in range(2)], 8) * np.tile([-.25, .25], 8)
            statistics = []
            for head in range(8):
                values = result[task, :, :, :, head].reshape(16, -1)
                estimate = comparison_ci([(values, weights)])
                rows.append((profile, name, head + 1, *estimate))
                statistics.append(estimate[0])
            print(profile, name, np.round(statistics, 3).tolist(), flush=True)
    write_table(args.output / 'decoder_l1_swaps.csv', ['profile', 'task', 'head', 'axis_contrast_db', 'ci_low', 'ci_high'], rows)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
