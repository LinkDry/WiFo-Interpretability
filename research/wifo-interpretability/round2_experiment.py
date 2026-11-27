"""Run WiFo Round 2: paired attention/value swaps and cumulative head masking."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from wifo_pilot import DISCOVERY, HEADS, TASKS, TRANSFER, Grid, Probe, TappedChannel
from wifo_pilot import as_tensor, atomic_npz, deltas, load_csi, prepared_input

AV_HEADS = (0, 1, 2, 4, 7)
AV_CONDITIONS = ('Identity', 'A-only', 'V-only', 'A+V')
FACTORS = ('doppler', 'delay')
BUDGETS = (.1, .2, .5)
METHODS = ('Task-Aware', 'Global Taylor', 'Same-Layer Random')
RANDOM_SEEDS = tuple(range(8701, 8711))
DECODER_LAYER = 6
TOL = dict(rtol=3e-5, atol=1e-6)


def make_probe(som, device):
    torch.manual_seed(71)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    return Probe(som, device)


@contextmanager
def av_site(probe, *, capture=None, head=None, donor_a=None, donor_v=None, batch_swaps=None):
    attention = probe.blocks[DECODER_LAYER].attn
    original = attention.forward

    def forward(x):
        count, tokens, width = x.shape
        q, k, values = (layer(x).reshape(count, tokens, 8, width // 8).transpose(1, 2)
                        for layer in (attention.q, attention.k, attention.v))
        maps = ((q @ k.transpose(-2, -1)) * attention.scale).softmax(-1)
        if capture is not None:
            capture.append((maps, values))
        if head is not None:
            if donor_a is not None:
                maps[:, head] = donor_a
            if donor_v is not None:
                values[:, head] = donor_v
        if batch_swaps is not None:
            for index, (swap_head, substitute_a, substitute_v) in enumerate(batch_swaps):
                rows = slice(index * len(substitute_a), (index + 1) * len(substitute_a))
                mode = index % len(AV_CONDITIONS)
                if mode in (0, 1, 3):
                    maps[rows, swap_head] = substitute_a
                if mode in (0, 2, 3):
                    values[rows, swap_head] = substitute_v
        output = (maps @ values).transpose(1, 2).reshape(count, tokens, width)
        return attention.proj_drop(attention.proj(output))

    attention.forward = forward
    try:
        yield
    finally:
        attention.forward = original


def tv_distance(first, second, queries):
    return .5 * (first[:, :, queries] - second[:, :, queries]).abs().sum(-1).mean(-1)


def swap_specs(recipient, donor):
    return [(head, (recipient if mode == 0 else donor)[0][:, head],
                   (recipient if mode == 0 else donor)[1][:, head])
            for head in AV_HEADS for mode in range(len(AV_CONDITIONS))]


@torch.inference_mode()
def run_av_profile(probe, source, batch):
    channel = TappedChannel(*(source[key] for key in ('path_gains', 'steering', 'doppler_relative', 'delay_relative', 'path_power')))
    csi = channel.render()
    grid = Grid((16, 32, 32))
    task_errors, task_drift, task_base = [], [], []
    max_identity_diff = 0.0
    for task in TASKS:
        errors, drift, bases = [], [], []
        mask = ~grid.visible(task, probe.device)
        for start in range(0, len(csi[0]), batch):
            clean = [as_tensor(condition[start:start + batch], probe) for condition in csi]
            received = [prepared_input(condition, task, torch.Generator(device=probe.device).manual_seed(8515 + start))
                        for condition in clean]
            targets = [probe.target(condition) for condition in clean]
            captured, baselines = [], []
            for observation, target in zip(received, targets):
                true_base = probe.error(observation, target, task)
                with av_site(probe, capture=captured):
                    measured = probe.error(observation, target, task)
                torch.testing.assert_close(true_base, measured, **TOL)
                baselines.append(true_base)
            bases.append(torch.stack(baselines).cpu().numpy())
            batch_errors, batch_drift = [], []
            for axis in (1, 2):
                cell_errors, cell_drift = [], []
                for cell in range(4):
                    original, alternative = captured[cell], captured[cell ^ axis]
                    specs = swap_specs(original, alternative)
                    size = len(received[cell])
                    with av_site(probe, batch_swaps=specs):
                        returned = probe.error(received[cell].repeat(len(specs), 1, 1, 1, 1),
                                               targets[cell].repeat(len(specs), 1, 1), task)
                    returned = returned.reshape(len(AV_HEADS), len(AV_CONDITIONS), size)
                    control = baselines[cell].expand_as(returned[:, 0])
                    torch.testing.assert_close(returned[:, 0], control, **TOL)
                    max_identity_diff = max(max_identity_diff, (returned[:, 0] - control).abs().max().item())
                    cell_errors.append(returned.cpu().numpy())
                    drift_heads = tv_distance(original[0][:, AV_HEADS], alternative[0][:, AV_HEADS], mask)
                    cell_drift.append(drift_heads.T.cpu().numpy())
                batch_errors.append(np.stack(cell_errors))
                batch_drift.append(np.stack(cell_drift))
            errors.append(np.stack(batch_errors))
            drift.append(np.stack(batch_drift))
        task_errors.append(np.concatenate(errors, axis=-1))
        task_drift.append(np.concatenate(drift, axis=-1))
        task_base.append(np.concatenate(bases, axis=-1))
    return dict(errors=np.stack(task_errors), drift=np.stack(task_drift), baseline=np.stack(task_base),
                heads=AV_HEADS, identity_max_abs_nmse_diff=max_identity_diff)


def run_av(args, probe, config):
    for profile in config['physics_seeds']:
        with np.load(args.pilot / f'physics_{profile}.npz') as source:
            result = run_av_profile(probe, source, config['physics_batch'])
        with np.load(args.pilot / f'decoder_l1_{profile}.npz') as audit:
            original = audit['errors'][:, :, :, 1, AV_HEADS, :]
            np.testing.assert_allclose(result['errors'][:, :, :, :, 1], original, **TOL)
            np.testing.assert_allclose(result['baseline'], audit['errors'][:, 0, :, 0, 0], **TOL)
        atomic_npz(args.output / f'av_{profile}.npz', **result)
        print(f"{profile} A/V complete, largest identity NMSE difference {result['identity_max_abs_nmse_diff']:.2g}", flush=True)


def csi_for(args, label, indices):
    root = args.som / 'dataset' / 'RF-only_channel_dataset_for_WiFo' / 'dataset'
    return load_csi(root / label.split('_')[0] / 'X_test.mat', indices)


@torch.inference_mode()
def scan_task(probe, task, csi, variants, batch, chunk):
    generator = torch.Generator(device=probe.device).manual_seed(7549)
    outputs = []
    for start in range(0, len(csi), batch):
        clean = as_tensor(csi[start:start + batch], probe)
        observed = prepared_input(clean, task, generator)
        outputs.append(probe.variant_errors(observed, probe.target(clean), task, variants, chunk))
    return np.concatenate(outputs, axis=-1)


def taylor_scores(probe, task, csi, batch):
    generator = torch.Generator(device=probe.device).manual_seed(7549)
    raw_scores, base = [], []
    for start in range(0, len(csi), batch):
        clean = as_tensor(csi[start:start + batch], probe)
        observed = prepared_input(clean, task, generator)
        gates = {layer: observed.new_ones((len(clean), 8), requires_grad=True) for layer in range(10)}
        loss = probe.error(observed, probe.target(clean), task, gates)
        # Each gate belongs to exactly one sample, so the sum yields per-sample gradients.
        gradients = torch.autograd.grad(loss.sum(), tuple(gates.values()))
        raw_scores.append(torch.cat(gradients, dim=-1).detach().abs().cpu().numpy())
        base.append(loss.detach().cpu().numpy())
    raw, baseline = np.concatenate(raw_scores), np.concatenate(base)
    return raw.mean(axis=0, dtype=np.float64) / baseline.mean(dtype=np.float64), baseline


def lock_count(effects, budget):
    feasible = np.flatnonzero(np.all(effects <= budget, axis=0))
    if not len(feasible):
        raise RuntimeError('The zero-head baseline should always be feasible')
    return int(feasible[-1])


def layer_counts(group):
    return np.bincount([head[0] for head in group], minlength=10)


def build_blind_groups(task_ranks, taylor_rank, counts):
    groups = {}
    for index, task in enumerate(TASKS):
        groups[task] = {}
        for budget, count in zip(BUDGETS, counts[index]):
            task_group = task_ranks[index][:count]
            by_layer = layer_counts(task_group)
            random_groups = []
            for seed in RANDOM_SEEDS:
                rng = np.random.default_rng(seed)
                random_groups.append(tuple((layer, int(head)) for layer, n in enumerate(by_layer)
                                           for head in sorted(rng.choice(8, int(n), replace=False))))
            groups[task][str(budget)] = {METHODS[0]: [task_group], METHODS[1]: [taylor_rank[:count]], METHODS[2]: random_groups}
    return groups


def json_write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def calibration(args, probe, config):
    discovered = []
    validation = tuple(f'{domain}_rep' for domain in DISCOVERY)
    for label in DISCOVERY:
        with np.load(args.pilot / f'heads_{label}.npz') as raw:
            np.testing.assert_array_equal(raw['heads'], HEADS)
            discovered.append(deltas(raw['errors']))
    ranking_scores = np.mean(discovered, axis=0)
    task_ranks = [[HEADS[i] for i in np.argsort(scores, kind='stable')] for scores in ranking_scores]
    taylor = []
    curves = {}
    for label in (*DISCOVERY, *validation):
        with np.load(args.pilot / f'heads_{label}.npz') as raw:
            indices, previous_base = raw['indices'], raw['errors'][:, 0]
        csi = csi_for(args, label, indices)
        scores_by_task, taylor_by_task = [], []
        for task_index, task in enumerate(TASKS):
            prefix = [task_ranks[task_index][:length] for length in range(81)]
            scores = scan_task(probe, task, csi, prefix, config['batch'], args.chunk)
            np.testing.assert_allclose(scores[0], previous_base[task_index], **TOL)
            scores_by_task.append(scores)
            if label in DISCOVERY:
                score, baseline = taylor_scores(probe, task, csi, config['batch'])
                np.testing.assert_allclose(scores[0], baseline, **TOL)
                taylor_by_task.append(score)
            print(f'{label} {task} cumulative scan k=0–80 complete', flush=True)
        if label in DISCOVERY:
            taylor.append(taylor_by_task)
        scores_by_task = np.stack(scores_by_task)
        atomic_npz(args.output / f'curve_{label}.npz', errors=scores_by_task, indices=indices)
        curves[label] = 10 * np.log10(scores_by_task.mean(-1, dtype=np.float64)
                                      / scores_by_task[:, :1].mean(-1, dtype=np.float64))
    taylor_scores_by_task = np.mean(taylor, axis=0)
    taylor_rank = [HEADS[i] for i in np.argsort(taylor_scores_by_task.mean(axis=0), kind='stable')]
    held_effects = np.stack([curves[label] for label in validation])
    counts = [[lock_count(held_effects[:, task], budget) for budget in BUDGETS] for task in range(2)]
    plans = build_blind_groups(task_ranks, taylor_rank, counts)
    json_write(args.output / 'prune_lock.json', {'groups': plans, 'counts': dict(zip(TASKS, counts)),
               'task_rank': dict(zip(TASKS, task_ranks)), 'taylor_rank': taylor_rank,
               'task_score_db': ranking_scores.tolist(), 'taylor_relative_score': taylor_scores_by_task.tolist(),
               'budgets_db': BUDGETS, 'ranking_domains': DISCOVERY, 'validation_domains': validation,
               'random_seeds': RANDOM_SEEDS, 'checkpoint_sha256': config['checkpoint_sha256']})
    print('Locked head counts (0.1, 0.2, 0.5 dB):', dict(zip(TASKS, counts)), flush=True)


def read_plan(path):
    locked = json.loads(path.read_text(encoding='utf-8'))
    variants, index = [], []
    for task in TASKS:
        task_groups, task_index = [()], []
        all_groups = {(): 0}
        for budget in BUDGETS:
            for method in METHODS:
                rows = []
                for group in locked['groups'][task][str(budget)][method]:
                    key = tuple(sorted(tuple(head) for head in group))
                    if key not in all_groups:
                        all_groups[key] = len(task_groups)
                        task_groups.append(key)
                    rows.append(all_groups[key])
                task_index.append(rows)
        variants.append(task_groups)
        index.append(task_index)
    return variants, index


def blind_evaluation(args, probe, config):
    path = args.output / 'prune_lock.json'
    variants, index = read_plan(path)
    lock_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    for domain in TRANSFER:
        with np.load(args.pilot / f'groups_{domain}.npz') as raw:
            prior = raw['indices']
        available = np.setdiff1d(np.arange(1000), prior)
        seed = 26831 + int(domain[1:])
        indices = np.sort(np.random.default_rng(seed).choice(available, config['samples_per_split'], replace=False))
        assert not np.intersect1d(indices, prior).size
        csi = csi_for(args, domain, indices)
        for task, task_variants in zip(TASKS, variants):
            values = scan_task(probe, task, csi, task_variants, config['batch'], args.chunk)
            atomic_npz(args.output / f'blind_{domain}_{task}.npz', errors=values, indices=indices, lock_sha256=lock_sha256)
            print(f'{domain} {task} locked evaluation complete', flush=True)
    json_write(args.output / 'blind_index.json', {'axis': 'task, then budget, then method, then replicate',
               'variant_rows': index, 'task_variants': variants, 'lock_sha256': lock_sha256})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--som', type=Path, default=Path('E:/Projects/SoM'))
    parser.add_argument('--pilot', type=Path, default=here / 'results' / 'base_pilot')
    parser.add_argument('--output', type=Path, default=here / 'results' / 'round2_av_prune')
    parser.add_argument('--stage', choices=('all', 'av', 'calibrate', 'blind'), default='all')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--chunk', type=int, default=8)
    args = parser.parse_args()
    if args.chunk <= 0:
        parser.error('--chunk must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pilot / 'run.json').read_text(encoding='utf-8'))
    probe = make_probe(args.som, args.device)
    with probe.checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != config['checkpoint_sha256']:
            raise RuntimeError('Checkpoint differs from Base pilot')
    source = subprocess.check_output(['git', '-C', str(args.som / 'WiFo'), 'rev-parse', 'HEAD'], text=True).strip()
    if source != config['source_commit']:
        raise RuntimeError('Source revision differs from Base pilot')
    if args.stage in ('all', 'av'):
        run_av(args, probe, config)
    if args.stage in ('all', 'calibrate'):
        calibration(args, probe, config)
    if args.stage in ('all', 'blind'):
        blind_evaluation(args, probe, config)
    json_write(args.output / 'run.json', {'base_pilot': config, 'round2_torch': torch.__version__,
               'round2_numpy': np.__version__, 'av_heads': [head + 1 for head in AV_HEADS],
               'av_conditions': AV_CONDITIONS, 'factors': FACTORS, 'prune_variant_chunk': args.chunk,
               'seeds_for_blind_sampling': [26831 + int(domain[1:]) for domain in TRANSFER]})


if __name__ == '__main__':
    main()
