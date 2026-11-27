"""Create three two-column SPL figures from verified experiment tables."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
TASKS = ('temporal', 'fre')
TITLES = ('Temporal prediction', 'Frequency prediction')
WIDTH = 7.05
COLORS = ('#00747A', '#865179', '#BD6929')
DARK = '#3F4C56'


def records(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name):
    fig.savefig(output / f'{name}.pdf', dpi=300)
    fig.savefig(output / f'{name}.png', dpi=300)
    plt.close(fig)


def head_heatmap(pilot, output):
    rows = records(pilot / 'head_effects.csv')
    shape = (10, 8)
    layers = [f'Enc{i}' for i in range(1, 7)] + [f'Dec{i}' for i in range(1, 5)]
    head_order = [f'{prefix}H{j}' for prefix in [f'E{i}' for i in range(1, 7)] + [f'D{i}' for i in range(1, 5)]
                  for j in range(1, 9)]
    held = [{row['head']: row for row in rows if row['domain'] == domain} for domain in ('D7_rep', 'D16_rep')]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.72), sharey=True, layout='constrained')
    norm = SymLogNorm(linthresh=.06, linscale=.65, vmin=-.025, vmax=7.2)
    for ax, key, title in zip(axes, ('time_delta_db', 'frequency_delta_db'), TITLES):
        matrix = np.mean([[float(by_head[h][key]) for h in head_order] for by_head in held], axis=0).reshape(shape)
        image = ax.pcolormesh(np.arange(9), np.arange(11), matrix, cmap='YlGnBu', norm=norm,
                              shading='flat', edgecolors='#FFFFFF', linewidth=.30)
        ax.set(xlim=(0, 8), ylim=(10, 0), xticks=np.arange(8)+.5, xticklabels=np.arange(1, 9),
               yticks=np.arange(10)+.5, yticklabels=layers, xlabel='Head index', title=title)
        ax.tick_params(length=0, pad=2)
        ax.axhline(6, color=DARK, lw=.9)
        ax.set_aspect('equal', adjustable='box')
        for layer, head in zip(*np.where(matrix >= .5)):
            value = matrix[layer, head]
            ax.text(head+.5, layer+.5, f'{value:.1f}', size=6.1, va='center', ha='center',
                    color='white' if norm(value) > .68 else '#111827')
    bar = fig.colorbar(image, ax=axes, shrink=.86, aspect=19, pad=.03, ticks=[0, .05, .2, 1., 4., 7.])
    bar.solids.set_rasterized(False)
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    bar.set_label(r'Single-head $\Delta$NMSE (dB)')
    save(fig, output, 'fig1_head_heatmap')


def av_plot(round2, output):
    rows = records(round2 / 'av_swap_effects.csv') + records(output / 'cdl_av_swap_effects.csv')
    summary = {(r['profile'], r['task'], r['head'], r['condition']): np.array([float(r[x]) for x in ('estimate','ci95_low','ci95_high')])
               for r in rows if r['metric'] == 'aligned_minus_off_axis'}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.32), sharex=True, layout='constrained')
    for row, condition in enumerate(('A-only','V-only')):
        for col, (task, title) in enumerate(zip(TASKS,TITLES)):
            ax = axes[row,col]
            ax.axhline(0, color='#909AA2', lw=.7, zorder=0)
            for offset, profile, color, marker in zip((-.18, 0, .18), ('diffuse','clustered','CDL-C'), COLORS, ('o','s','D')):
                points = np.stack([summary[profile, task, f'Dec1-H{head}', condition] for head in range(1,4)]).T
                estimate, lo, hi = points
                x = np.arange(3)+offset
                ax.vlines(x, lo, hi, color=color, lw=1.0)
                ax.plot(x, estimate, marker=marker, ms=4, c=color, linestyle='none', label=profile.capitalize() if profile != 'CDL-C' else profile)
            ax.set(xlim=(-.47, 2.46), xticks=range(3), xticklabels=['H1', 'H2', 'H3'])
            if row==0:
                ax.set_title(title)
                ax.set_ylim(-.05, .24)
                ax.yaxis.set_major_locator(FixedLocator([-.05, 0, .1, .2]))
            else:
                ax.set_ylim(-2.45, 3.0)
                ax.yaxis.set_major_locator(FixedLocator([-2,-1,0,1,2,3]))
                ax.set_xlabel('First decoder layer head')
            if col==0:
                ax.set_ylabel(f'{condition}\nAxis contrast (dB)')
    axes[0,1].legend(loc='upper left', frameon=False, fontsize=6.4, labelspacing=.3, borderpad=.25)
    save(fig, output, 'fig2_av_decoupling')


def pruning_plot(round2, output):
    curve_rows = records(round2 / 'pruning_curve.csv')
    blind_rows = records(round2 / 'pruning_blind_eval.csv')
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.55), layout='constrained', gridspec_kw={'width_ratios':(1.08,.92)})
    for row, (task, title, count) in enumerate(zip(TASKS, TITLES, (16, 23))):
        ax = axes[row, 0]
        for domain, color in zip(('D7_rep','D16_rep'),COLORS):
            values = np.array([[int(x['pruned_heads']), *map(float, (x['estimate'], x['ci95_low'], x['ci95_high']))]
                               for x in curve_rows if x['task']==task and x['split']==domain]).T
            x, est, lo, hi = values
            ax.plot(x, est, color=color, lw=1.25, label=domain.split('_')[0])
            ax.fill_between(x, lo, hi, color=color, alpha=.13, lw=0)
        y = next(float(x['estimate']) for x in curve_rows if x['task']==task and x['split']=='Validation worst' and int(x['pruned_heads'])==count)
        ax.scatter(count, y, color=DARK, s=15, zorder=4)
        ax.axhline(.2, color='#87919A', ls='--', lw=.75)
        ax.set(yscale='symlog', ylabel=f'{title}\n$\\Delta$NMSE (dB)', xlim=(0,80), xticks=(0,20,40,60,80))
        ax.set_yscale('symlog', linthresh=.13)
        ax.yaxis.set_major_locator(FixedLocator([0, .2, 1, 10]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.set_ylim(-.075, 14)
        if row==0:
            ax.set_title('Joint masking on held-out samples')
            ax.legend(loc='upper left', frameon=False, fontsize=6.4, ncol=2, handlelength=1.2, columnspacing=.8)
        else:
            ax.set_xlabel('Number of masked heads')
        ax = axes[row, 1]
        methods = ('Task-Aware','Global Taylor','Same-Layer Random')
        for method_index,(method, color) in enumerate(zip(methods, COLORS)):
            bars = []
            for domain in ('D17','D18'):
                data = next(r for r in blind_rows if r['task']==task and r['budget_db']=='0.2' and r['method']==method and r['domain']==domain)
                bars.append([float(data[x]) for x in ('estimate','ci95_low','ci95_high')])
            center, low, high = np.stack(bars).T
            x = np.arange(2)+(method_index-1)*.22
            ax.bar(x, center, width=.19, color=color, label=('Task-aware','Global Taylor','Layer random')[method_index], zorder=2)
            ax.vlines(x, low, high, color=DARK, lw=.9, zorder=3)
            ax.hlines(low, x-.036, x+.036, color=DARK, lw=.7, zorder=3)
            ax.hlines(high, x-.036, x+.036, color=DARK, lw=.7, zorder=3)
        ax.axhline(.2, color='#87919A', ls='--', lw=.75)
        ax.set(xticks=range(2), xticklabels=['D17', 'D18'], xlim=(-.65,1.62), ylabel=r'$\Delta$NMSE (dB)')
        ax.set_title(f'{count} heads masked on unseen configurations')
        if row==0:
            ax.set_ylim(0, .57)
            ax.yaxis.set_major_locator(FixedLocator([0,.2,.4]))
            ax.legend(loc='upper center', frameon=False, fontsize=6.15, ncol=3, handlelength=.9, columnspacing=.7, handletextpad=.3)
        else:
            ax.set_ylim(-.095, 2.23)
            ax.yaxis.set_major_locator(FixedLocator([0,.2,1,2]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    save(fig, output, 'fig3_pruning_blind')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=ROOT/'results/base_pilot')
    parser.add_argument('--round2', type=Path, default=ROOT/'results/round2_av_prune')
    parser.add_argument('--output', type=Path, default=ROOT/'results/round3_spl_final')
    args = parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.3,'axes.titlesize':8.1,'axes.labelsize':7.4,
                         'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.55,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    head_heatmap(args.pilot, args.output)
    av_plot(args.round2, args.output)
    pruning_plot(args.round2, args.output)


if __name__=='__main__':
    main()
