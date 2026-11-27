"""Physical attention-head slicing and reproducible WiFo inference benchmarks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter_ns

import numpy as np
import torch
from torch import nn

from round2_experiment import make_probe
from round2_report import ci, log_draws
from wifo_pilot import TASKS, TRANSFER, Grid, as_tensor, atomic_npz, load_csi, prepared_input, write_table

BUDGETS = (0, 16, 23)
PRED_TOL = dict(atol=1e-5, rtol=3e-5)
NMSE_TOL = dict(atol=1e-6, rtol=3e-5)


class CutAttention(nn.Module):
    def __init__(self, original, keep):
        super().__init__()
        self.num_heads = len(keep)
        self.head_dim = original.q.out_features // original.num_heads
        self.scale = original.scale
        width = self.num_heads * self.head_dim
        weight = original.q.weight
        channels = (torch.tensor(keep, dtype=torch.long, device=weight.device)[:, None] * self.head_dim
                    + torch.arange(self.head_dim, device=weight.device)).flatten()
        factory = {'device': weight.device, 'dtype': weight.dtype}
        for name in ('q', 'k', 'v'):
            old = getattr(original, name)
            new = nn.Linear(old.in_features, width, bias=old.bias is not None, **factory)
            with torch.no_grad():
                new.weight.copy_(old.weight[channels])
                if new.bias is not None:
                    new.bias.copy_(old.bias[channels])
            setattr(self, name, new)
        self.proj = nn.Linear(width, original.proj.out_features, bias=original.proj.bias is not None, **factory)
        with torch.no_grad():
            self.proj.weight.copy_(original.proj.weight[:, channels])
            if self.proj.bias is not None:
                self.proj.bias.copy_(original.proj.bias)
        self.proj_drop = original.proj_drop

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = [projection(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
                   for projection in (self.q, self.k, self.v)]
        weights = ((q @ k.transpose(-2, -1)) * self.scale).softmax(-1)
        merged = (weights @ v).transpose(1, 2).reshape(batch, tokens, self.proj.in_features)
        return self.proj_drop(self.proj(merged))


def probe_at(som, device):
    return make_probe(som, device)


def build_submodel(som, device, plan, budget):
    probe = probe_at(som, device)
    state = json.loads(plan.read_text(encoding='utf-8'))
    with probe.checkpoint.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != state['checkpoint_sha256']:
            raise ValueError('WiFo checkpoint differs from the frozen Round 2 checkpoint')
    removed = tuple(map(tuple, state['taylor_rank'][:budget]))
    assert len(set(removed)) == budget
    removed_by_layer = defaultdict(set)
    for layer, head in removed:
        removed_by_layer[layer].add(head)
    for layer, heads in removed_by_layer.items():
        block = probe.blocks[layer]
        keep = [head for head in range(block.attn.num_heads) if head not in heads]
        block.attn = CutAttention(block.attn, keep)
    probe.model.eval().requires_grad_(False)
    return probe, removed


def predict(probe, signal, task, gates=None):
    probe.active_gates = gates
    try:
        scale = [1, 1, 1]
        latent, mask, order, shape = probe.model.forward_encoder(signal, .5, task, scale=scale)
        decoded = probe.model.forward_decoder(latent, order, task, input_size=shape, scale=scale)
        return probe.model.decoder_pred(decoded), mask
    finally:
        probe.active_gates = None


def paired_nmse(pred, target, missing):
    result = torch.complex(*pred[:, missing].chunk(2, dim=-1))
    reference = target[:, missing]
    return (result - reference).abs().square().sum((1, 2)) / reference.abs().square().sum((1, 2))


def gate_mask(sample, removed):
    masks = {}
    for layer, head in removed:
        masks.setdefault(layer, sample.new_ones((len(sample), 8)))[:, head] = 0
    return masks


def data_path(som, domain):
    return som / 'dataset/RF-only_channel_dataset_for_WiFo/dataset' / domain / 'X_test.mat'


@torch.inference_mode()
def check_blind(som, device, round2, output, batch):
    rows = []
    maximum = 0.
    plan = round2 / 'prune_lock.json'
    lock_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    reference, _ = build_submodel(som, device, plan, 0)
    models = {count: build_submodel(som, device, plan, count) for count in BUDGETS[1:]}
    lookup = json.loads((round2 / 'blind_index.json').read_text(encoding='utf-8'))
    assert lookup['lock_sha256'] == lock_sha
    for domain in TRANSFER:
        for task_index, task in enumerate(TASKS):
            with np.load(round2 / f'blind_{domain}_{task}.npz') as raw:
                assert raw['lock_sha256'].item() == lock_sha
                indices, reference_scores = raw['indices'], raw['errors']
            values = load_csi(data_path(som, domain), indices)
            generator = torch.Generator(device=reference.device).manual_seed(7549)
            full_scores = []
            structured = {count: [] for count in models}
            logical = {count: [] for count in models}
            max_by_count = dict.fromkeys(models, 0.)
            for offset in range(0, len(indices), batch):
                clean = as_tensor(values[offset:offset + batch], reference)
                sample = prepared_input(clean, task, generator)
                target = reference.target(clean)
                missing = ~Grid(tuple(clean.shape[2:])).visible(task, clean.device)
                baseline, mask = predict(reference, sample, task)
                assert torch.equal(mask.bool(), missing.expand_as(mask))
                full_scores.append(paired_nmse(baseline, target, missing).cpu().numpy())
                for count, (compact, removed) in models.items():
                    gated, _ = predict(reference, sample, task, gate_mask(sample, removed))
                    sliced, _ = predict(compact, sample, task)
                    torch.testing.assert_close(sliced, gated, **PRED_TOL)
                    max_by_count[count] = max(max_by_count[count], (sliced - gated).abs().max().item())
                    structured[count].append(paired_nmse(sliced, target, missing).cpu().numpy())
                    logical[count].append(paired_nmse(gated, target, missing).cpu().numpy())
            full = np.concatenate(full_scores)
            np.testing.assert_allclose(full, reference_scores[0], **NMSE_TOL)
            domains_seed = 683107 + TRANSFER.index(domain)
            for count in models:
                a = np.concatenate(structured[count])
                b = np.concatenate(logical[count])
                np.testing.assert_allclose(a, b, **NMSE_TOL)
                same_task_r2 = count == (16, 23)[task_index]
                round2_difference = ''
                if same_task_r2:
                    # Budget index 1, Global Taylor index 1 among the three methods.
                    round2_row = lookup['variant_rows'][task_index][4][0]
                    np.testing.assert_allclose(a, reference_scores[round2_row], **NMSE_TOL)
                    round2_difference = float(np.abs(a - reference_scores[round2_row]).max())
                draws = log_draws(np.stack((full, a, b)), domains_seed)
                rows.append([count, domain, task, len(indices), *ci(draws[1]-draws[0]),
                             *ci(draws[2]-draws[0]), max_by_count[count], round2_difference])
                maximum = max(maximum, max_by_count[count])
                atomic_npz(output / f'blind_k{count}_{domain}_{task}.npz', indices=indices, base=full,
                           structured=a, logical=b, lock_sha256=lock_sha)
    write_table(output / 'structured_blind.csv', ['removed', 'domain', 'task', 'n', 'delta_nmse_db', 'ci95_low', 'ci95_high',
                'gate_delta_db', 'gate_ci95_low', 'gate_ci95_high', 'max_abs_prediction_difference',
                'max_abs_nmse_difference_from_round2_when_available'], rows)
    print(f'Structured equivalence complete. Maximum absolute prediction difference {maximum:.4g}.', flush=True)


def estimate_gflops(probe, tensor, task):
    macs = 0
    hooks = []

    def linear(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    def conv(module, inputs, output):
        nonlocal macs
        macs += output.numel() * (module.in_channels // module.groups) * int(np.prod(module.kernel_size))

    def attention(module, inputs, output):
        nonlocal macs
        batch, tokens, _ = inputs[0].shape
        macs += 2 * batch * tokens**2 * module.q.out_features

    for module in probe.model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear))
        elif isinstance(module, nn.Conv3d):
            hooks.append(module.register_forward_hook(conv))
    for block in probe.blocks:
        hooks.append(block.attn.register_forward_hook(attention))
    try:
        predict(probe, tensor, task)
    finally:
        for hook in hooks:
            hook.remove()
    return 2 * macs / 1e9


def inference_probe(som, device, plan, removed):
    probe, _ = build_submodel(som, device, plan, removed)
    for hook in probe.handles:
        hook.remove()
    return probe


@torch.inference_mode()
def benchmark(som, device, round2, output):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('The end-to-end benchmark requires CUDA')
    with np.load(round2 / 'blind_D17_temporal.npz') as raw:
        indices = raw['indices'][:16]
    inputs = load_csi(data_path(som, 'D17'), indices)
    metrics = {}
    plan = round2 / 'prune_lock.json'
    # Warm both kernels and the persistent CUDA library workspace before memory accounting.
    initializer = inference_probe(som, device, plan, 0)
    input_for_init = as_tensor(inputs, initializer)
    for task in TASKS:
        for size in (1, 16):
            predict(initializer, input_for_init[:size], task)
    torch.cuda.synchronize(device)
    del initializer, input_for_init
    gc.collect()
    torch.cuda.empty_cache()
    idle_allocated = torch.cuda.memory_allocated(device)
    for count in BUDGETS:
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated(device) == idle_allocated, 'Previous model is still allocated'
        probe = inference_probe(som, device, plan, count)
        for task in TASKS:
            for batch_size in (1, 16):
                tensor = as_tensor(inputs[:batch_size], probe)
                tensor = prepared_input(tensor, task, torch.Generator(device=device).manual_seed(781))
                for _ in range(5):
                    predict(probe, tensor, task)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                mem_before = torch.cuda.memory_allocated(device)
                predict(probe, tensor, task)
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                gflops = estimate_gflops(probe, tensor, task)
                params = sum(p.numel() for p in probe.model.parameters())
                metrics[count, task, batch_size] = [gflops, params, peak / 2**20, (peak-mem_before)/2**20]
                del tensor
        del probe
    results, samples = [], []
    active = {count: inference_probe(som, device, plan, count) for count in BUDGETS}
    rng = np.random.default_rng(93278)
    for task in TASKS:
        for batch_size in (1, 16):
            tensor = as_tensor(inputs[:batch_size], active[0])
            tensor = prepared_input(tensor, task, torch.Generator(device=device).manual_seed(781))
            for _ in range(30):
                for probe in active.values():
                    predict(probe, tensor, task)
            numbers = {count: [] for count in BUDGETS}
            for repeat in range(50):
                for order, count in enumerate(rng.permutation(BUDGETS), start=1):
                    torch.cuda.synchronize(device)
                    started = perf_counter_ns()
                    prediction = predict(active[count], tensor, task)
                    torch.cuda.synchronize(device)
                    duration_ms = (perf_counter_ns() - started) / 1e6
                    numbers[count].append(duration_ms)
                    samples.append([count, task, batch_size, repeat + 1, order, duration_ms])
                    del prediction
            for count in BUDGETS:
                p50, p95 = np.percentile(numbers[count], [50, 95])
                results.append([count, 80-count, task, batch_size, p50, p95, *metrics[count, task, batch_size]])
                print(f'{count:2} removed {task:8} batch={batch_size:2} p50={p50:.3f} ms p95={p95:.3f} ms', flush=True)
    write_table(output / 'table_iii_benchmark.csv', ['removed', 'retained', 'task', 'batch', 'p50_ms', 'p95_ms',
                'gflops_per_batch', 'parameters', 'peak_allocated_mib', 'incremental_peak_mib'], results)
    write_table(output / 'latency_trials.csv', ['removed', 'task', 'batch', 'trial', 'order_within_trial', 'ms'], samples)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    metadata = {'device': props.name, 'driver': driver, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'precision': 'FP32 with TF32 disabled', 'timed_samples': 50, 'warmup': 30,
                'shape': [16, 32, 32], 'timing': 'synchronized wall clock from GPU-resident visible CSI to token predictions',
                'order': 'All three models are resident during timing. Each round randomly permutes their order, seed 93278.',
                'gflops': '2 FLOPs per MAC for linear, patch Conv3d, QK and AV matmul',
                'memory': 'CUDA peak allocated memory including one model, its input and forward intermediates',
                'persistent_cuda_workspace_mib': idle_allocated / 2**20,
                'indices': indices.tolist()}
    (output / 'benchmark_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--som', type=Path, default=Path('E:/Projects/SoM'))
    parser.add_argument('--round2', type=Path, default=here / 'results/round2_av_prune')
    parser.add_argument('--output', type=Path, default=here / 'results/round3_spl_final')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('all', 'blind', 'benchmark'), default='all')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in ('all', 'blind'):
        check_blind(args.som, args.device, args.round2, args.output, batch=16)
    if args.stage in ('all', 'benchmark'):
        benchmark(args.som, args.device, args.round2, args.output)


if __name__ == '__main__':
    main()
