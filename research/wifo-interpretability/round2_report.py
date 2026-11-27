"""Create Round 2 tables, figures, and a report from saved per-sample scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from round2_experiment import AV_CONDITIONS, AV_HEADS, BUDGETS, FACTORS, METHODS, RANDOM_SEEDS, TOL, layer_counts, lock_count, read_plan
from wifo_pilot import DISCOVERY, TASKS, TRANSFER, write_table

N_BOOTSTRAP = 2000
PROFILES = ('diffuse', 'clustered')
VALIDATION = ('D7_rep', 'D16_rep')
STAT_COLUMNS = ['estimate', 'ci95_low', 'ci95_high']


def resampling_weights(size, seed):
    generator = np.random.default_rng(seed)
    weights = generator.multinomial(size, np.full(size, 1. / size), size=N_BOOTSTRAP) / size
    return np.vstack((np.full(size, 1. / size), weights)).T


def mean_draws(values, seed):
    return values.astype(np.float64, copy=False) @ resampling_weights(values.shape[-1], seed)


def ci(draws):
    return [float(draws[0]), *map(float, np.quantile(draws[1:], [.025, .975]))]


def interval_points(axis, x, center, low, high, color, *, marker='o', label=None):
    axis.vlines(x, low, high, color=color, linewidth=1.2)
    axis.plot(x, center, linestyle='none', marker=marker, markersize=4, color=color, label=label)


def log_draws(values, seed):
    return 10 * np.log10(mean_draws(values, seed))


def effects_by_metric(values, task):
    return (values[0], values[1], values[task] - values[1 - task])


def summarize_av(output):
    av_rows, drift_rows, av_lookup, drift_lookup, identity = [], [], {}, {}, {}
    for profile_index, profile in enumerate(PROFILES):
        with np.load(output / f'av_{profile}.npz') as source:
            errors, drift, base = source['errors'], source['drift'], source['baseline']
            samples = errors.shape[-1]
            assert errors.shape == (2, 2, 4, 5, 4, samples) and drift.shape == (2, 2, 4, 5, samples)
            assert np.isfinite(errors).all() and np.all(errors > 0)
            assert np.isfinite(drift).all() and np.all((0 <= drift) & (drift <= 1))
            np.testing.assert_array_equal(source['heads'], AV_HEADS)
            control = np.broadcast_to(base[:, None, :, None, :], errors[..., 0, :].shape)
            np.testing.assert_allclose(errors[..., 0, :], control, **TOL)
            identity[profile] = float(source['identity_max_abs_nmse_diff'])
        seed = 118037 + profile_index
        delta = (log_draws(errors, seed) - log_draws(base, seed)[:, None, :, None, None, :]).mean(axis=2)
        drift_draws = mean_draws(drift, seed).mean(axis=2)
        for index, task in enumerate(TASKS):
            for head_index, head in enumerate(AV_HEADS):
                name = f'Dec1-H{head + 1}'
                d = delta[index, :, head_index, :]
                conditions = [*AV_CONDITIONS, 'V-only minus A-only', 'A x V interaction']
                comparisons = [*d.transpose(1, 0, 2), d[:, 2] - d[:, 1], d[:, 3] - d[:, 1] - d[:, 2] + d[:, 0]]
                for condition, values in zip(conditions, comparisons):
                    for metric, draws in zip((*FACTORS, 'aligned_minus_off_axis'), effects_by_metric(values, index)):
                        estimate = ci(draws)
                        av_rows.append([profile, task, name, condition, metric, samples, *estimate])
                        av_lookup[profile, task, name, condition, metric] = estimate
                for metric, draws in zip((*FACTORS, 'aligned_minus_off_axis'), effects_by_metric(drift_draws[index, :, head_index], index)):
                    estimate = ci(draws)
                    drift_rows.append([profile, task, name, metric, samples, *estimate])
                    drift_lookup[profile, task, name, metric] = estimate
    write_table(output / 'av_swap_effects.csv', ['profile', 'task', 'head', 'condition', 'metric', 'n_pairs', *STAT_COLUMNS], av_rows)
    write_table(output / 'attn_drift.csv', ['profile', 'task', 'head', 'metric', 'n_pairs', *STAT_COLUMNS], drift_rows)
    return av_lookup, drift_lookup, identity


def summarize_curves(output, plan):
    rows, lookup, all_draws = [], {}, {}
    for domain_index, domain in enumerate((*DISCOVERY, *VALIDATION)):
        with np.load(output / f'curve_{domain}.npz') as source:
            data = source['errors']
            assert data.shape[1] == 81 and np.isfinite(data).all() and np.all(data > 0)
            assert len(np.unique(source['indices'])) == data.shape[-1]
            draws = log_draws(data, 6112 + domain_index)
        all_draws[domain] = draws - draws[:, :1]
    all_draws['Validation worst'] = np.maximum.reduce([all_draws[domain] for domain in VALIDATION])
    for index, task in enumerate(TASKS):
        for budget, count in zip(BUDGETS, plan['counts'][task]):
            assert count == lock_count(all_draws['Validation worst'][index, :, 0][None, :], budget)
        for count in range(81):
            layers = layer_counts(plan['task_rank'][task][:count])
            distribution = [int(layers[0]), int(layers[1:6].sum()), int(layers[6]), int(layers[7:].sum())]
            for label, data in all_draws.items():
                row = [*ci(data[index, count]), *distribution]
                rows.append([task, label, count, *row])
                lookup[task, label, count] = row
    write_table(output / 'pruning_curve.csv', ['task', 'split', 'pruned_heads', *STAT_COLUMNS,
                'enc1', 'enc2_to_6', 'dec1', 'dec2_to_4'], rows)
    return lookup


def bootstrap_seeds(draws, weights):
    return np.concatenate(([draws[:, 0].mean()], np.einsum('sr,rs->r', draws[:, 1:], weights)))


def summarize_blind(output, plan):
    expected_sha = hashlib.sha256((output / 'prune_lock.json').read_bytes()).hexdigest()
    variants, index = read_plan(output / 'prune_lock.json')
    mapping = json.loads((output / 'blind_index.json').read_text(encoding='utf-8'))
    assert expected_sha == mapping['lock_sha256'] and index == mapping['variant_rows']
    rows, lookup = [], {}
    for task_index, task in enumerate(TASKS):
        domain_effects = []
        for domain_index, domain in enumerate(TRANSFER):
            with np.load(output / f'blind_{domain}_{task}.npz') as source:
                assert source['lock_sha256'].item() == expected_sha
                data = source['errors']
                assert data.shape[0] == len(variants[task_index]) and np.isfinite(data).all() and np.all(data > 0)
                draws = log_draws(data, 907381 + domain_index)
            domain_effects.append(draws - draws[:1])
        rng = np.random.default_rng(91354)
        n_seeds = len(RANDOM_SEEDS)
        seed_weights = rng.multinomial(n_seeds, np.ones(n_seeds) / n_seeds, size=N_BOOTSTRAP) / n_seeds
        for budget_index, (budget, count) in enumerate(zip(BUDGETS, plan['counts'][task])):
            reference = None
            for method_index, method in enumerate(METHODS):
                ids = index[task_index][budget_index * len(METHODS) + method_index]
                raw = np.stack([effects[ids] for effects in domain_effects])
                if method == METHODS[2]:
                    assert len(ids) == n_seeds
                    dist = layer_counts(plan['groups'][task][str(budget)][METHODS[0]][0])
                    for group in plan['groups'][task][str(budget)][method]:
                        np.testing.assert_array_equal(layer_counts(group), dist)
                    estimates = np.stack([bootstrap_seeds(domain, seed_weights) for domain in raw])
                else:
                    estimates = raw[:, 0]
                by_split = np.concatenate((estimates, estimates.max(axis=0, keepdims=True)), axis=0)
                if method_index == 0:
                    reference = by_split
                seed_points = raw[..., 0]
                seed_by_split = np.concatenate((seed_points, seed_points.max(axis=0, keepdims=True)), axis=0)
                for split, effect, contrast, seed_effect in zip((*TRANSFER, 'Worst domain'), by_split, by_split - reference, seed_by_split):
                    stats = [*ci(effect), *ci(contrast), float(seed_effect.min()), float(seed_effect.max())]
                    rows.append([task, budget, count, method, split, len(ids), *stats])
                    lookup[task, budget, method, split] = stats
    write_table(output / 'pruning_blind_eval.csv', ['task', 'budget_db', 'pruned_heads', 'method', 'domain',
                'n_seeds', *STAT_COLUMNS, 'minus_task_aware_db', 'minus_task_aware_ci95_low', 'minus_task_aware_ci95_high',
                'seed_min_db', 'seed_max_db'], rows)
    return lookup


COLORS = ('#087E83', '#8B4573', '#AF611F')


def plot_av(output, av, drift):
    conditions = AV_CONDITIONS[1:]
    numbers = [head + 1 for head in AV_HEADS]
    for profile in PROFILES:
        fig, panels = plt.subplots(2, 3, figsize=(12.2, 6.2), layout='constrained', sharex=True)
        for row, task in enumerate(TASKS):
            label = 'Time prediction' if task == 'temporal' else 'Frequency prediction'
            for col, metric in enumerate(FACTORS):
                ax = panels[row, col]
                ax.axhline(0, color='#8E98A5', linewidth=.7)
                for offset, condition, color in zip((-.19, 0, .19), conditions, COLORS):
                    middle, low, high = np.array([av[profile, task, f'Dec1-H{head}', condition, metric] for head in numbers]).T
                    interval_points(ax, np.arange(len(numbers)) + offset, middle, low, high, color, label=condition)
                if row == 0:
                    ax.set_title(f'{metric.capitalize()} swap')
                if col == 0:
                    ax.set_ylabel(f'{label}\nΔNMSE (dB)')
            ax = panels[row, 2]
            for offset, metric, color in zip((-.11, .11), FACTORS, ('#30618B', '#A66A20')):
                middle, low, high = np.array([drift[profile, task, f'Dec1-H{head}', metric] for head in numbers]).T
                interval_points(ax, np.arange(len(numbers)) + offset, middle, low, high, color, label=metric.capitalize())
            if row == 0:
                ax.set_title('Attention-map drift')
            ax.set_ylabel('Total variation')
        for ax in panels[1]:
            ax.set_xticks(range(len(numbers)), [f'H{i}' for i in numbers])
            ax.set_xlabel('First decoder layer')
        panels[0, 0].legend(frameon=False, fontsize=8)
        panels[0, 2].legend(frameon=False, fontsize=8)
        fig.suptitle(f'{profile.capitalize()} synthetic multipath · paired 95% bootstrap intervals', fontsize=13)
        fig.savefig(output / f'av_{profile}.png', dpi=185)
        plt.close(fig)


def plot_pruning(output, plan, curves, blind):
    fig, panels = plt.subplots(2, 2, figsize=(11.6, 7.3), layout='constrained')
    for row, task in enumerate(TASKS):
        label = 'Time prediction' if task == 'temporal' else 'Frequency prediction'
        ax = panels[row, 0]
        for domain, color in zip(VALIDATION, COLORS):
            values = np.array([curves[task, domain, count][:3] for count in range(81)])
            ax.plot(np.arange(81), values[:, 0], color=color, lw=1.55, label=domain.replace('_rep', ' held-out'))
            ax.fill_between(np.arange(81), values[:, 1], values[:, 2], color=color, alpha=.11, lw=0)
        ax.set_yscale('symlog', linthresh=.15)
        ax.axhline(.2, color='#626D7B', ls='--', lw=.8, label='0.2 dB budget')
        count = plan['counts'][task][1]
        ax.scatter(count, curves[task, 'Validation worst', count][0], s=36, color='#212433', zorder=4)
        ax.set(xlabel='Number of jointly masked heads', ylabel=f'{label}\nΔNMSE (dB)', xlim=(0, 80))
        if row == 0:
            ax.set_title('Held-out cumulative pruning curve')
            ax.legend(frameon=False, fontsize=8)
        ax = panels[row, 1]
        for method, color in zip(METHODS, COLORS):
            values = np.array([blind[task, budget, method, 'Worst domain'][:3] for budget in BUDGETS])
            xx = np.array(plan['counts'][task])
            ax.plot(xx, values[:, 0], color=color, lw=1.2)
            interval_points(ax, xx, *values.T, color, label=method)
        ax.axhline(.2, color='#626D7B', ls='--', lw=.8)
        ax.set(xlabel='Matched number of masked heads', ylabel='Worst-domain ΔNMSE (dB)')
        if row == 0:
            ax.legend(frameon=False, fontsize=8)
            ax.set_title('Locked evaluation on D17 and D18')
    fig.savefig(output / 'pruning_av_blind.png', dpi=200)
    plt.close(fig)


def db(stats):
    return f'{stats[0]:+.3f} [{stats[1]:+.3f}, {stats[2]:+.3f}]'


def head_id(layer_head):
    layer, head = layer_head
    module = f'Enc{layer + 1}' if layer < 6 else f'Dec{layer - 5}'
    return f'{module}-H{head + 1}'


def write_report(path, plan, av, drift, identity, curves, blind):
    text = ['# WiFo Base Round 2 实验报告', '', '2026-11-27', '',
            '冻结 WiFo v1 Base，在第一层解码器测试注意力图与 Value 的配对反事实替换，同时在各剪枝预算上测量联合头消融。所有区间为 2,000 次配对 bootstrap 的 95% 区间。', '',
            '## 第一层解码器的注意力图与 Value 替换', '',
            '沿用两套 Base pilot 合成多径，每套 128 个配对信道，RMS 多普勒 40/140 Hz × RMS 时延扩展 50/200 ns。每个受体与供体只差一个物理因素，掩码和标准噪声抽样沿用原协议。按四个受体条件平均各自的 NMSE 变化（dB），时域轴选择性为多普勒代价减时延代价，频域反之。', '',
            f'Identity 与原始推理的最大逐样本线性 NMSE 差为 {max(identity.values()):.2g}。同样本 A-only 也通过 Round 1 原始结果的数值回归检查。', '',
            '下表给出各头 **V-only − A-only 的轴选择性差**。正值表示在本次交换中，V-only 的轴对比度更高。', '',
            '| 多径 | 任务 | H1 | H2 | H3 | H5 | H8 |', '|---|---|---:|---:|---:|---:|---:|']
    for profile in PROFILES:
        for task in TASKS:
            cells = [db(av[profile, task, f'Dec1-H{n + 1}', 'V-only minus A-only', 'aligned_minus_off_axis']) for n in AV_HEADS]
            text.append(f'| {profile} | {task} | ' + ' | '.join(cells) + ' |')
    text.extend(['', 'H2 的时域轴选择性主要体现于 Value 交换。H3 在频域任务中的 Attention 交换对时延更敏感，而 Value 交换对多普勒更敏感，两套多径均复现该方向。', '',
                 '| 多径 | 任务与头 | 条件 | 多普勒 ΔNMSE | 时延 ΔNMSE | 轴选择性 |',
                 '|---|---|---|---:|---:|---:|'])
    for profile in PROFILES:
        for task, head in [('temporal', 'Dec1-H2'), ('fre', 'Dec1-H3')]:
            for mode in AV_CONDITIONS[1:]:
                cells = [db(av[profile, task, head, mode, factor]) for factor in (*FACTORS, 'aligned_minus_off_axis')]
                text.append(f'| {profile} | {task} {head} | {mode} | ' + ' | '.join(cells) + ' |')
    h8_values = [drift[profile, 'fre', 'Dec1-H8', factor][0] for profile in PROFILES for factor in FACTORS]
    text.extend(['', f'H8 在频域任务中的注意力图总变差点估计为 {min(h8_values):.3f}–{max(h8_values):.3f}，支持较小幅度的配对变化。该距离对待预测查询行计算，纳入全部键。物理因素变换的幅度沿用 pilot，因此比较对应于所选水平与交换位置。', '',
                 '[A/V 完整效应表](results/round2_av_prune/av_swap_effects.csv) 含所有头的多普勒代价、时延代价、轴选择性和 dB 尺度的 A×V 交互。[注意力图漂移表](results/round2_av_prune/attn_drift.csv) 含两轴的距离及其差值。', '',
                 '## 联合头消融与阈值锁定', '',
                 '两项任务分别按 D7/D16 发现集的平均单头 ΔNMSE 升序排列。整条前缀曲线覆盖所有整数头数 0–80，以 D7/D16 留出集点估计的最坏域退化锁定满足预算的最大头数。表中的验证区间固定已选头组，选择过程使用点估计。Global Taylor 用逐样本的门控梯度绝对值，除以对应域和任务的平均线性基线 NMSE，再对发现域与两项任务等权平均。Same-Layer Random 每项用 10 个预先锁定的随机种子，逐层数量与 Task-Aware 一致。', '',
                 '| 任务 | 容限 | 头数 | Enc1 / Enc2–6 / Dec1 / Dec2–4 | 留出集最坏 ΔNMSE |', '|---|---:|---:|---|---:|'])
    for task in TASKS:
        for budget, count in zip(BUDGETS, plan['counts'][task]):
            result = curves[task, 'Validation worst', count]
            text.append(f'| {task} | {budget:.1f} dB | {count}/80 | ' + ' / '.join(map(str, result[3:])) + f' | {db(result)} |')
    text.extend(['', '锁定头组后，D17、D18 各从 Round 1 未使用的索引中再取 160 个样本。最坏域取两个域的 ΔNMSE 最大值。同层随机法先取种子均值，再取最坏域。其区间对种子和域内样本共同重采样，两个域使用同一组抽中的种子。', '',
                 '| 任务 | 容限 | 头数 | 方法 | D17 ΔNMSE | D18 ΔNMSE | 最坏域 ΔNMSE |', '|---|---:|---:|---|---:|---:|---:|'])
    for task in TASKS:
        budget = .2
        for method in METHODS:
            cells = [db(blind[task, budget, method, domain]) for domain in (*TRANSFER, 'Worst domain')]
            text.append(f"| {task} | {budget:.1f} dB | {plan['counts'][task][1]} | {method} | " + ' | '.join(cells) + ' |')
    text.append('')
    for task in TASKS:
        worst = blind[task, .2, METHODS[0], 'Worst domain']
        relation = '低于或等于' if worst[0] <= .2 else '高于'
        text.append(f'- `{task}` 在 0.2 dB 验证预算下锁定 **{plan["counts"][task][1]}/80** 个头，盲测最坏域退化 **{worst[0]:.3f} dB**，点估计{relation} 0.2 dB。')
    taylor_time = blind['temporal', .2, METHODS[1], 'Worst domain'][3:6]
    taylor_freq = blind['fre', .2, METHODS[1], 'Worst domain'][3:6]
    random_freq = blind['fre', .2, METHODS[2], 'Worst domain'][3:6]
    text.append(f'- 在相同头数下，Global Taylor 相对 Task-Aware 的最坏域退化差为时域 {db(taylor_time)} dB、频域 {db(taylor_freq)} dB。频域 Same-Layer Random 相对 Task-Aware 的差为 {db(random_freq)} dB。')
    text.extend(['', '参考预算的盲测结果及相对 Task-Aware 的配对差见 [盲测结果表](results/round2_av_prune/pruning_blind_eval.csv)。[锁定的头组与随机种子](results/round2_av_prune/prune_lock.json) 和 [完整联合消融曲线](results/round2_av_prune/pruning_curve.csv) 可用于复核。以上头数均基于运行时联合屏蔽。', '',
                 '## 结果图', '',
                 '[分散多径的 A/V 交换](results/round2_av_prune/av_diffuse.png) · [延迟簇多径的 A/V 交换](results/round2_av_prune/av_clustered.png) · [联合消融曲线和盲测](results/round2_av_prune/pruning_av_blind.png)', '',
                 '## 复现', '', '```powershell', 'python -m unittest -v test_round2_experiment test_wifo_pilot',
                 'python round2_experiment.py', 'python round2_report.py', '```', ''])
    path.write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--output', type=Path, default=root / 'results' / 'round2_av_prune')
    parser.add_argument('--report', type=Path, default=root / 'ROUND2_REPORT.md')
    args = parser.parse_args()
    output = args.output
    plan = json.loads((output / 'prune_lock.json').read_text(encoding='utf-8'))
    av, drift, identity = summarize_av(output)
    curves = summarize_curves(output, plan)
    blind = summarize_blind(output, plan)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    plot_av(output, av, drift)
    plot_pruning(output, plan, curves, blind)
    write_report(args.report, plan, av, drift, identity, curves, blind)
    print('Identity max. absolute differences', identity)
    print('Locked heads', plan['counts'])
    for task in TASKS:
        for method in METHODS:
            print(task, method, db(blind[task, .2, method, 'Worst domain']))


if __name__ == '__main__':
    main()
