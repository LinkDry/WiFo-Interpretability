"""Summarize an existing pilot and generate its diagnostic figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import spearmanr

from wifo_pilot import DISCOVERY, HEADS, REPLICATION, TASKS, TRANSFER, comparison_ci, deltas, write_table


def display_head(head):
    layer, index = head
    prefix = f'E{layer + 1}' if layer < 6 else f'D{layer - 5}'
    return f'{prefix}H{index + 1}'


def contrast(array, first, second):
    values = np.stack((array[0, first], array[0, second], array[1, first], array[1, second]))
    return values, [1, -1, -1, 1]


def plot_heads(output, discovery, replication, a, b):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelsize': 11, 'axes.titlesize': 13, 'savefig.bbox': 'tight'})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4), layout='constrained')
    for idx, delta, title in ((0, discovery, 'Discovery · D7 + D16'), (1, replication, 'Independent samples · D7 + D16')):
        panel = (ax, ax2)[idx]
        panel.scatter(*delta, color='#A9AEBA', s=22, alpha=.75, lw=0)
        for group, color, marker, label in ((a, '#17757C', '^', 'Time candidates'), (b, '#B04766', 's', 'Frequency candidates')):
            indexes = [HEADS.index(tuple(head)) for head in group]
            panel.scatter(*delta[:, indexes], color=color, marker=marker, s=56, lw=.8, edgecolor='white', label=label)
        lower, upper = min(0, discovery.min(), replication.min()), max(discovery.max(), replication.max())
        pad = .05 * (upper - lower)
        panel.plot([lower - pad, upper + pad], [lower - pad, upper + pad], color='#718096', lw=1, ls='--')
        panel.set(xlim=(lower-pad, upper+pad), ylim=(lower-pad, upper+pad), title=title,
                  xlabel='Time prediction ΔNMSE (dB)', ylabel='Frequency prediction ΔNMSE (dB)', aspect='equal')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.savefig(output / 'head_scan.png', dpi=220)
    plt.close(fig)


def plot_physics(output, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
    for ax, (profile, summary) in zip(axes, summaries.items()):
        for index, item in enumerate(summary['aligned_minus_off_axis_swap']):
            midpoint, low, high = item['ci']
            color = ('#17757C' if item['task'] == 'temporal' else '#B04766')
            ax.errorbar(midpoint, index, xerr=[[midpoint - low], [high - midpoint]], fmt='o', color=color,
                        markersize=6, capsize=3, lw=1.5)
        ax.axvline(0, color='#8E98A5', lw=1, ls='--')
        ax.set(title=f'{profile.capitalize()} paths', xlabel='Aligned − off-axis map swap ΔNMSE (dB)')
        ax.xaxis.set_major_locator(MaxNLocator(6))
    axes[0].set_yticks(range(4), ['Time · candidate', 'Time · same-layer control', 'Frequency · candidate', 'Frequency · same-layer control'])
    axes[0].invert_yaxis()
    fig.suptitle('Paired synthetic multipath · 95% bootstrap intervals', fontsize=12)
    fig.savefig(output / 'physics_swap.png', dpi=220)
    plt.close(fig)


def analyze(output):
    cfg = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assignments = json.loads((output / 'groups.json').read_text(encoding='utf-8'))
    groups = assignments['groups']
    controls = assignments['controls']
    labels = (*DISCOVERY, *REPLICATION, *TRANSFER)
    singleton = {label: np.load(output / f'heads_{label}.npz') for label in (*DISCOVERY, *REPLICATION)}
    grouped = {label: np.load(output / f'groups_{label}.npz') for label in labels}
    for label in labels:
        data = grouped[label]
        assert data['errors'].shape == (2, 27, cfg['samples_per_split'])
        assert np.isfinite(data['errors']).all() and np.all(data['errors'] > 0)
        if label in singleton:
            assert np.array_equal(data['indices'], singleton[label]['indices'])
            assert np.array_equal(singleton[label]['heads'], HEADS)
            np.testing.assert_allclose(data['errors'][:, 0], singleton[label]['errors'][:, 0], rtol=3e-5, atol=1e-6)
    for label in DISCOVERY:
        assert not np.intersect1d(singleton[label]['indices'], singleton[label + '_rep']['indices']).size
    individual = {label: deltas(data['errors']) for label, data in singleton.items()}
    first = np.stack([individual[label] for label in DISCOVERY]).mean(0)
    held = np.stack([individual[label] for label in REPLICATION[:2]]).mean(0)
    ranks = {label: float(spearmanr(first[0] - first[1], result[0] - result[1]).statistic)
             for label, result in individual.items() if label in REPLICATION}
    ranks['same_domains_held_mean'] = float(spearmanr(first[0] - first[1], held[0] - held[1]).statistic)
    write_table(output / 'head_effects.csv', ['domain', 'head', 'time_delta_db', 'frequency_delta_db', 'time_minus_frequency_db'],
                [(label, display_head(head), d[0, i], d[1, i], d[0, i] - d[1, i])
                 for label, d in individual.items() for i, head in enumerate(HEADS)])
    group_stats = {}
    for label, data in grouped.items():
        errors = data['errors']
        by_task = {}
        for index, task in enumerate(TASKS):
            own = index + 1
            other = 2 - index
            center, lo, hi = comparison_ci([(errors[index, [own, other]], [1, -1])])
            effects = deltas(errors)[index]
            own_controls = effects[2 + 12 * index:2 + 12 * (index + 1)]
            by_task[task] = {'time_group_delta': float(effects[0]), 'frequency_group_delta': float(effects[1]),
                             'own_minus_other_ci': [center, lo, hi],
                             'matched_random_median': float(np.median(own_controls)),
                             'matched_random_exceeded': int((own_controls < effects[index]).sum()),
                             'baseline_db': float(10 * np.log10(errors[index, 0].mean()))}
        group_stats[label] = {'double_dissociation_ci': comparison_ci([contrast(errors, 1, 2)]), 'tasks': by_task}
    group_stats['held_same_domains_mean'] = {'double_dissociation_ci': comparison_ci([
        (values, np.asarray(coefficients) / 2) for values, coefficients in (
            contrast(grouped[label]['errors'], 1, 2) for label in REPLICATION[:2])])}
    individual_ci = {}
    for label, dataset in singleton.items():
        values = dataset['errors']
        individual_ci[label] = {}
        for head in (tuple(groups[0][0]), tuple(groups[1][0])):
            row = HEADS.index(head) + 1
            individual_ci[label][display_head(head)] = {
                task: comparison_ci([(values[index, [0, row]], [-1, 1])])
                for index, task in enumerate(TASKS)}
    physics = {}
    phys_rows = []
    for file in sorted(output.glob('physics_*.npz')):
        profile = file.stem.split('_')[1]
        source = np.load(file)
        errors, spans, swaps, head_order = [source[name] for name in ('errors', 'spans', 'swaps', 'heads')]
        for array in (errors, spans, swaps):
            assert np.isfinite(array).all()
        assert errors.shape == (2, 4, 5, cfg['samples_per_profile'])
        for axis in ('doppler', 'delay'):
            relative, power = source[f'{axis}_relative'], source['path_power']
            np.testing.assert_allclose(np.sqrt((power * (relative - (power * relative).sum(-1, keepdims=True)) ** 2).sum(-1)),
                                       1., rtol=1e-10)
        phys_controls = [controls[0], controls[12]]
        axes = [[0, 1, 0, 1], [0, 0, 1, 1]]
        span_rows, swap_rows, ablation_rows, extra = [], [], [], []
        for task, task_name in enumerate(TASKS):
            for group, kind, heads in ((task, 'candidate', groups[task]), (task + 2, 'same_layer', phys_controls[task])):
                indices = [head_order.tolist().index(head) for head in heads]
                aligned_axis = task
                off_axis = 1 - task
                weights = [np.array(axis) - .5 for axis in axes]
                # Average over the other factor and over both direction swaps.
                replacement = swaps[task, group].reshape(2, 8, -1)
                ci = comparison_ci([(replacement.reshape(16, -1),
                                     np.repeat([1 if aligned_axis == i else -1 for i in range(2)], 8) * np.tile([-.25, .25], 8))])
                swap_rows.append({'task': task_name, 'group': kind, 'ci': ci})
                factor_swaps = [comparison_ci([(replacement[axis], np.tile([-.25, .25], 4))]) for axis in range(2)]
                # Match the physical factor to the direction of the measured lag.
                span_for_task = spans[task][:, indices].mean(1)  # condition, sample, lag axis, units
                y = span_for_task[:, :, :, 1]
                off_change = np.einsum('i,ij->j', weights[off_axis], y[:, :, aligned_axis])
                on_change = np.einsum('i,ij->j', weights[aligned_axis], y[:, :, aligned_axis])
                rng = np.random.default_rng(442)
                bootstrap_idx = rng.integers(0, len(on_change), size=(2000, len(on_change)))
                def summarize_change(values):
                    return [float(values.mean()), *map(float, np.quantile(values[bootstrap_idx].mean(1), [.025, .975]))]
                span_rows.append({'task': task_name, 'group': kind, 'aligned_span_ci': summarize_change(on_change),
                                  'off_axis_change_in_same_span_ci': summarize_change(off_change),
                                  'aligned_minus_off_span_ci': summarize_change(on_change - off_change)})
                env_effects = errors[task][:, [0, group + 1], :].reshape(8, -1)
                on_off = weights[aligned_axis] - weights[off_axis]
                ablation = comparison_ci([(env_effects, np.repeat(on_off, 2) * np.tile([-1, 1], 4))])
                ablation_rows.append({'task': task_name, 'group': kind, 'aligned_minus_off_axis_change_ci': ablation})
                extra.append({'task': task_name, 'group': kind, 'doppler_swap_ci': factor_swaps[0], 'delay_swap_ci': factor_swaps[1]})
                phys_rows.append((profile, task_name, kind, *ci, *span_rows[-1]['aligned_span_ci'], *ablation))
        control_contrasts = []
        for task, task_name in enumerate(TASKS):
            candidate = swaps[task, task].reshape(16, -1)
            control = swaps[task, task + 2].reshape(16, -1)
            other_candidate = swaps[task, 1 - task].reshape(16, -1)
            coefficients = np.repeat([1 if task == i else -1 for i in range(2)], 8) * np.tile([-.25, .25], 8)
            contrast_weights = np.concatenate([coefficients, -coefficients])
            ci = comparison_ci([(np.concatenate([candidate, control]), contrast_weights)])
            versus_other = comparison_ci([(np.concatenate([candidate, other_candidate]), contrast_weights)])
            control_contrasts.append({'task': task_name, 'ci': ci, 'versus_other_candidate_ci': versus_other})
        physics[profile] = {'aligned_minus_off_axis_swap': swap_rows, 'normalized_spans': span_rows,
                            'ablation_interaction': ablation_rows, 'swap_by_factor': extra,
                            'candidate_minus_control_swap_selectivity': control_contrasts,
                            'baseline_db_by_task_and_condition': (10 * np.log10(errors[:, :, 0].mean(-1))).tolist()}
    result = {'groups': [[display_head(head) for head in group] for group in groups], 'head_rank_spearman': ranks,
              'top_single_head_effects': individual_ci,
              'group_results': group_stats, 'physics': physics}
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_table(output / 'physics_effects.csv', ['profile', 'task', 'group', 'swap_axis_diff_db', 'swap_ci_low', 'swap_ci_high',
                'aligned_norm_span_change', 'span_ci_low', 'span_ci_high', 'ablation_axis_diff_db', 'ablation_ci_low', 'ablation_ci_high'], phys_rows)
    plot_heads(output, first, held, *groups)
    plot_physics(output, physics)
    print('groups', result['groups'])
    print('head ranks', ranks)
    for label, stats in group_stats.items():
        print(label, np.round(stats['double_dissociation_ci'], 3),
              {k: (round(v['time_group_delta'], 3), round(v['frequency_group_delta'], 3)) for k, v in stats.get('tasks', {}).items()})
    for name, data in physics.items():
        print(name, 'swap selectivity over control', [(entry['task'], np.round(entry['ci'], 3)) for entry in data['candidate_minus_control_swap_selectivity']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
