"""Frozen WiFo Base pilot. All paths and effect definitions are recorded with the run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import spearmanr

TASKS = ('temporal', 'fre')
DISCOVERY = ('D7', 'D16')
REPLICATION = ('D7_rep', 'D16_rep', 'D11')
TRANSFER = ('D17', 'D18')
HEADS = tuple((layer, head) for layer in range(10) for head in range(8))
FACTORS = ((40., 50.), (140., 50.), (40., 200.), (140., 200.))  # Hz and ns
PROFILE_SEEDS = {'diffuse': 861, 'clustered': 2381}


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]

    def coordinates(self, device=None):
        return torch.cartesian_prod(*(torch.arange(n // 4, device=device) for n in self.shape))

    def visible(self, task, device=None):
        axis = TASKS.index(task) * 2
        return self.coordinates(device)[:, axis] < self.shape[axis] // 8

    def lag_matrices(self, task, encoder, device):
        coords = self.coordinates(device).float()
        visible = self.visible(task, device)
        keys = coords[visible]
        queries = keys if encoder else coords[~visible]
        distances = (queries[:, None, :] - keys[None, :, :]).abs()
        # Time patches are 2 ms apart and frequency patches are 720 kHz apart.
        return distances[..., (0, 2)] * distances.new_tensor([2., 720.])


def gate_input(channels, gates):
    split = channels.unflatten(-1, (gates.shape[-1], -1))
    return (split * gates[:, None, :, None]).flatten(-2)


@dataclass(frozen=True)
class TappedChannel:
    gains: np.ndarray
    spatial: np.ndarray
    doppler: np.ndarray
    delays: np.ndarray
    power: np.ndarray

    def weighted_spread(self, values):
        center = (self.power * values).sum(axis=1, keepdims=True)
        return np.sqrt((self.power * (values - center) ** 2).sum(axis=1))

    @classmethod
    def draw(cls, count, seed, profile):
        rng = np.random.default_rng(seed)
        paths = 18 if profile == 'diffuse' else 12
        delays = rng.exponential(size=(count, paths))
        if profile == 'clustered':
            delays = (.15 * delays + np.tile([0., 1., 2.], paths // 3))
        delays -= delays.min(axis=1, keepdims=True)
        doppler = np.cos(rng.uniform(0, 2 * np.pi, size=(count, paths)))
        weight = rng.lognormal(0., .55, size=(count, paths)) * np.exp(-.45 * delays)
        power = weight / weight.sum(axis=1, keepdims=True)
        gains = np.sqrt(power) * np.exp(2j * np.pi * rng.random((count, paths)))
        wavelength_coords = np.stack(np.meshgrid(np.arange(4), np.arange(8), indexing='ij'), -1).reshape(32, 2)
        azimuth = rng.uniform(-np.pi, np.pi, (count, paths))
        elevation = rng.uniform(-.6, .6, (count, paths))
        direction = np.stack((np.sin(azimuth) * np.cos(elevation), np.sin(elevation)), -1)
        spatial = np.exp(1j * np.pi * np.einsum('npa,ka->npk', direction, wavelength_coords))
        raw = cls(gains, spatial, doppler, delays, power)
        doppler -= (power * doppler).sum(axis=1, keepdims=True)
        return cls(gains, spatial, doppler / raw.weighted_spread(doppler)[:, None],
                   delays / raw.weighted_spread(delays)[:, None], power)

    def render(self):
        times = .0005 * np.arange(16)
        frequencies = 180e3 * np.arange(32)
        outputs = []
        for doppler_hz, delay_ns in FACTORS:
            phase_t = np.exp(2j * np.pi * self.doppler[..., None] * doppler_hz * times)
            phase_f = np.exp(-2j * np.pi * self.delays[..., None] * delay_ns * 1e-9 * frequencies)
            csi = np.einsum('np,npt,npa,npf->ntaf', self.gains, phase_t, self.spatial, phase_f, optimize=True)
            outputs.append(np.stack([csi.real, csi.imag], axis=1).astype('float32'))
        return outputs


def comparison_ci(terms, repeats=2000):
    """Paired per-domain bootstrap of a weighted sum of log mean NMSEs."""
    rng = np.random.default_rng(5093)
    estimate = 0.
    draws = np.zeros(repeats)
    for values, coefficient in terms:
        values = np.asarray(values, dtype=np.float64)
        coefficient = np.asarray(coefficient)
        indices = rng.integers(0, values.shape[-1], size=(repeats, values.shape[-1]))
        estimate += np.dot(10 * np.log10(values.mean(axis=-1)), coefficient)
        draws += np.dot(coefficient, 10 * np.log10(values[:, indices].mean(axis=-1)))
    return [float(estimate), *map(float, np.quantile(draws, [.025, .975]))]


def load_csi(path, indices):
    if h5py.is_hdf5(path):
        with h5py.File(path) as file:
            data = file['X_val'][..., indices.tolist()].transpose(3, 2, 1, 0)
            real, imag = data['real'], data['imag']
    else:
        data = loadmat(path, variable_names=['X_val'])['X_val'][indices]
        real, imag = data.real, data.imag
    return np.stack((real, imag), axis=1).astype('float32')


def prepared_input(csi, task, gen):
    shape = csi.shape[2:]
    axis = TASKS.index(task) * 2
    # Choose the same prefix as the public patch-level mask.
    extent = (shape[axis] // 8) * 4
    selector = [slice(None)] * csi.ndim
    selector[axis + 2] = slice(0, extent)
    visible = csi[tuple(selector)]
    power = visible.square().mean(dim=(1, 2, 3, 4), keepdim=True)
    standard_noise = torch.randn(csi.shape, device=csi.device, generator=gen)
    out = torch.zeros_like(csi)
    out[tuple(selector)] = visible + standard_noise[tuple(selector)] * (.01 * power).sqrt()
    return out


class Probe:
    def __init__(self, root, device):
        self.device = torch.device(device)
        source = root / 'WiFo' / 'src'
        sys.path.insert(0, str(source))
        from model import WiFo_model
        options = SimpleNamespace(size='base', t_patch_size=4, patch_size=4, pos_emb='SinCos_3D', no_qkv_bias=0)
        self.model = WiFo_model(options).eval().to(self.device)
        self.checkpoint = root / 'WiFo' / 'WiFo' / 'wifo_base.pkl'
        weights = torch.load(self.checkpoint, weights_only=True, map_location='cpu')
        self.model.load_state_dict({k: v for k, v in weights.items() if not k.startswith('decoder_pred_dual.')}, strict=True)
        self.blocks = tuple(self.model.blocks) + tuple(self.model.decoder_blocks)
        self.model.requires_grad_(False)
        self.active_gates = None
        self.handles = []
        for layer, block in enumerate(self.blocks):
            def hook(module, inputs, layer=layer):
                if self.active_gates is not None and layer in self.active_gates:
                    return (gate_input(inputs[0], self.active_gates[layer]),)
                return None
            self.handles.append(block.attn.proj.register_forward_pre_hook(hook))

    def target(self, csi):
        return self.model.patchify(csi)

    def error(self, data, target, task, gates=None):
        self.active_gates = gates
        try:
            scale = [1, 1, 1]
            latent, mask, order, shape = self.model.forward_encoder(data, .5, task, scale=scale)
            decoded = self.model.forward_decoder(latent, order, task, input_size=shape, scale=scale)
            pred = self.model.decoder_pred(decoded)
            miss = ~Grid(tuple(data.shape[2:])).visible(task, data.device)
            if not torch.equal(mask.bool(), miss.expand_as(mask)):
                raise RuntimeError('Unexpected WiFo token order or mask')
            pred = torch.complex(*pred[:, miss].chunk(2, dim=-1))
            true = target[:, miss]
            return (pred - true).abs().square().sum((1, 2)) / true.abs().square().sum((1, 2))
        finally:
            self.active_gates = None

    def variant_errors(self, data, target, task, variants, chunk):
        rows = []
        count = len(data)
        for start in range(0, len(variants), chunk):
            groups = variants[start:start + chunk]
            gates = {}
            for index, group in enumerate(groups):
                for layer, head in group:
                    if layer not in gates:
                        gates[layer] = data.new_ones((len(groups) * count, 8))
                    gates[layer][index * count:(index + 1) * count, head] = 0
            rows.append(self.error(data.repeat(len(groups), 1, 1, 1, 1),
                                   target.repeat(len(groups), 1, 1), task, gates).reshape(len(groups), count))
        return torch.cat(rows).cpu().numpy()

    @contextmanager
    def attention(self, group, task, grid, capture=None, donor=None, spans=None):
        saved = []
        layers = sorted(set(layer for layer, _ in group))
        for layer in layers:
            mod = self.blocks[layer].attn
            heads = [head for idx, head in group if idx == layer]
            local_distances = grid.lag_matrices(task, layer < 6, self.device)
            visible = grid.visible(task, self.device)
            original = mod.forward

            def replacement(x, mod=mod, layer=layer, heads=heads, distances=local_distances):
                count, tokens, width = x.shape
                q, k, v = [p(x).reshape(count, tokens, 8, width // 8).transpose(1, 2) for p in (mod.q, mod.k, mod.v)]
                weights = ((q @ k.transpose(-2, -1)) * mod.scale).softmax(-1)
                if capture is not None:
                    capture[layer] = weights[:, heads].clone()
                if donor is not None:
                    weights[:, heads] = donor[layer]
                if spans is not None:
                    picked = weights[:, heads]
                    if layer >= 6:
                        picked = picked[:, :, ~visible][:, :, :, visible]
                    mass = picked.sum(dim=-1).clamp_min(1e-12)
                    expected = torch.einsum('bhqk,qka->bhqa', picked, distances) / mass[..., None]
                    reference = distances.mean(dim=1)
                    normalized = expected / reference.clamp_min(1e-12)
                    spans[layer] = torch.stack((expected.mean(dim=2), normalized.mean(dim=2)), dim=-1).cpu().numpy()
                values = (weights @ v).transpose(1, 2).reshape(count, tokens, width)
                return mod.proj_drop(mod.proj(values))

            mod.forward = replacement
            saved.append((mod, original))
        try:
            yield
        finally:
            for mod, original in saved:
                mod.forward = original


def atomic_npz(path, **values):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def as_tensor(array, probe):
    return torch.as_tensor(np.ascontiguousarray(array), device=probe.device)


def scan(probe, csi, variants, batch, chunk):
    outputs = []
    for task in TASKS:
        generator = torch.Generator(device=probe.device).manual_seed(7549)
        task_results = []
        for start in range(0, len(csi), batch):
            clean = as_tensor(csi[start:start + batch], probe)
            prepared = prepared_input(clean, task, generator)
            task_results.append(probe.variant_errors(prepared, probe.target(clean), task, variants, chunk))
        outputs.append(np.concatenate(task_results, axis=-1))
    return np.stack(outputs)


def dataset_indices(n):
    splits = {}
    for label in (*DISCOVERY, 'D11', *TRANSFER):
        permutation = np.random.default_rng(19841 + int(label[1:])).permutation(1000)
        splits[label] = np.sort(permutation[:n])
        if label in DISCOVERY:
            splits[f'{label}_rep'] = np.sort(permutation[n:2 * n])
    return splits


def deltas(scores):
    return 10 * np.log10(scores[:, 1:].mean(axis=-1, dtype=np.float64) / scores[:, :1].mean(axis=-1, dtype=np.float64))


def candidate_groups(scores, count=4):
    scores = np.stack([deltas(scores[label]) for label in DISCOVERY]).mean(axis=0)
    ranking = np.argsort(scores[0] - scores[1], kind='stable')
    return tuple(tuple(HEADS[index] for index in order) for order in (ranking[-count:][::-1], ranking[:count]))


def matched_controls(groups, count=12):
    rng = np.random.default_rng(8119)
    controls = []
    for group in groups:
        for _ in range(count):
            control = []
            for layer, _ in group:
                excluded = set(groups[0] + groups[1] + tuple(control))
                pool = [head for head in range(8) if (layer, head) not in excluded]
                if not pool:
                    pool = [head for head in range(8) if (layer, head) not in set(group + tuple(control))]
                control.append((layer, int(rng.choice(pool))))
            controls.append(tuple(control))
    return tuple(controls)


def run_phy(probe, csi, groups, control_groups, batch):
    recorded = []
    variants = [(), *groups, *control_groups]
    observed = tuple(dict.fromkeys(groups[0] + groups[1] + sum(tuple(control_groups), ())))
    grid = Grid((16, 32, 32))
    for task in TASKS:
        losses = []
        span_data = []
        replacements = []
        for start in range(0, len(csi[0]), batch):
            clean = [as_tensor(array[start:start + batch], probe) for array in csi]
            data = [prepared_input(arr, task, torch.Generator(device=probe.device).manual_seed(8515 + start)) for arr in clean]
            targets = [probe.target(arr) for arr in clean]
            losses.append(np.stack([probe.variant_errors(item, tgt, task, variants, 4) for item, tgt in zip(data, targets)]))
            batch_spans = []
            for item, tgt in zip(data, targets):
                collector = {}
                with probe.attention(observed, task, grid, spans=collector):
                    observed_err = probe.error(item, tgt, task)
                base = probe.error(item, tgt, task)
                torch.testing.assert_close(observed_err, base, rtol=2e-5, atol=1e-6)
                head_spans = []
                for layer, head in observed:
                    layer_heads = [h for block, h in observed if block == layer]
                    head_spans.append(collector[layer][:, layer_heads.index(head)])
                batch_spans.append(np.stack(head_spans))
            span_data.append(np.stack(batch_spans))
            # Hold the recipient values fixed at each replacement site. The donor
            # changes by exactly one physical factor, with the other at either level.
            swaps = []
            for group in (*groups, *control_groups):
                donors = []
                originals = []
                for item, tgt in zip(data, targets):
                    captured = {}
                    with probe.attention(group, task, grid, capture=captured):
                        originals.append(probe.error(item, tgt, task))
                    donors.append(captured)
                for index in (0, 3):
                    with probe.attention(group, task, grid, donor=donors[index]):
                        restored = probe.error(data[index], targets[index], task)
                    torch.testing.assert_close(restored, originals[index], rtol=2e-5, atol=1e-6)
                swap_by_factor = []
                for delta in (1, 2):
                    swap = []
                    for recipient in range(4):
                        with probe.attention(group, task, grid, donor=donors[recipient ^ delta]):
                            error = probe.error(data[recipient], targets[recipient], task)
                        swap.append(torch.stack((originals[recipient], error)).cpu().numpy())
                    swap_by_factor.append(np.stack(swap))
                swaps.append(np.stack(swap_by_factor))
            replacements.append(np.stack(swaps))
        recorded.append((np.concatenate(losses, axis=-1), np.concatenate(span_data, axis=2), np.concatenate(replacements, axis=-1)))
    losses, spans, swaps = (np.stack([entry[index] for entry in recorded]) for index in range(3))
    return losses, spans, swaps, np.array(observed)


def write_table(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(args):
    root, output = args.som.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run.json').exists():
        raise FileExistsError(f'Choose a fresh run folder: {output}')
    torch.manual_seed(71)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    probe = Probe(root, args.device)
    data_root = root / 'dataset' / 'RF-only_channel_dataset_for_WiFo' / 'dataset'
    splits = dataset_indices(args.samples)
    raw = {}
    singleton = [(), *((head,) for head in HEADS)]
    sentinel = load_csi(data_root / 'D16' / 'X_test.mat', splits['D16'][:2])
    for task in TASKS:
        clean = as_tensor(sentinel, probe)
        held = prepared_input(clean, task, torch.Generator(device=probe.device).manual_seed(71))
        target = probe.target(clean)
        ordinary = probe.error(held, target, task)
        torch.testing.assert_close(ordinary, probe.error(held, target, task, {0: held.new_ones((len(held), 8))}), rtol=0, atol=0)
        parallel = probe.variant_errors(held, target, task, [(), ((0, 0),)], 2)
        torch.testing.assert_close(ordinary.cpu(), torch.from_numpy(parallel[0]), rtol=1e-5, atol=1e-6)
        if torch.allclose(ordinary.cpu(), torch.from_numpy(parallel[1])):
            raise AssertionError('Single-head gate has no observable effect')
    for label in (*DISCOVERY, *REPLICATION):
        path = data_root / label.split('_')[0] / 'X_test.mat'
        raw[label] = scan(probe, load_csi(path, splits[label]), singleton, args.batch, args.chunk)
        atomic_npz(output / f'heads_{label}.npz', errors=raw[label], indices=splits[label], heads=HEADS)
        print('head scan:', label, raw[label].shape, flush=True)
    groups = candidate_groups(raw)
    controls = matched_controls(groups)
    print('locked groups:', groups, flush=True)
    variants = [(), *groups, *controls]
    (output / 'groups.json').write_text(json.dumps({'groups': groups, 'controls': controls}, indent=2), encoding='utf-8')
    for label in (*DISCOVERY, *REPLICATION, *TRANSFER):
        path = data_root / label.split('_')[0] / 'X_test.mat'
        scores = scan(probe, load_csi(path, splits[label]), variants, args.batch, args.chunk)
        atomic_npz(output / f'groups_{label}.npz', errors=scores, indices=splits[label])
        print('group tests:', label, scores.shape, flush=True)
    # Use the first layer-matched random group of each kind, fixed by seed, for the more expensive map swaps.
    phy_controls = (controls[0], controls[len(controls) // 2])
    for profile, seed in PROFILE_SEEDS.items():
        paths = TappedChannel.draw(args.physics_samples, seed, profile)
        losses, spans, swaps, heads = run_phy(probe, paths.render(), groups, phy_controls, args.physics_batch)
        atomic_npz(output / f'physics_{profile}.npz', errors=losses, spans=spans, swaps=swaps, heads=heads,
                   factors=FACTORS, doppler_relative=paths.doppler, delay_relative=paths.delays,
                   path_gains=paths.gains, path_power=paths.power, steering=paths.spatial)
        print('physics:', profile, losses.shape, spans.shape, swaps.shape, flush=True)
    meta = {'experiment': 'WiFo Base exact ablations and paired 2x2 controlled multipath',
            'checkpoint_sha256': hashlib.file_digest(probe.checkpoint.open('rb'), 'sha256').hexdigest(),
            'source_commit': subprocess.check_output(['git', '-C', str(root / 'WiFo'), 'rev-parse', 'HEAD'], text=True).strip(),
            'device': str(probe.device), 'gpu': torch.cuda.get_device_name(probe.device) if probe.device.type == 'cuda' else '',
            'torch': torch.__version__, 'numpy': np.__version__, 'tasks': TASKS, 'samples_per_split': args.samples,
            'samples_per_profile': args.physics_samples, 'batch': args.batch, 'variant_chunk': args.chunk,
            'physics_batch': args.physics_batch, 'physics_seeds': PROFILE_SEEDS, 'axis_order': 'N,C,T,A,F',
            'snr_db': 20, 'mask_fraction': .5, 'target': 'clean masked CSI', 'physics_factors_hz_ns': FACTORS}
    (output / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--som', type=Path, default=Path('E:/Projects/SoM'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples', type=int, default=160)
    parser.add_argument('--physics-samples', type=int, default=128)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=4)
    parser.add_argument('--physics-batch', type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.samples <= 500 or min(args.batch, args.chunk, args.physics_batch) < 1 or args.physics_samples < 2:
        parser.error('Require 2–500 samples per split, at least two synthetic pairs, and positive batch sizes')
    with torch.inference_mode():
        run(args)


if __name__ == '__main__':
    main()
