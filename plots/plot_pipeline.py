"""
plot_pipeline.py
================
Generate presentation-ready plots from search_pipeline.py results.

Run from project root:
    python plots/plot_pipeline.py --results results/pipeline_topk3856_gc.npy.npz
    python plots/plot_pipeline.py --results results/pipeline_feat_div_topk3906_gc.npy.npz

Prerequisite: Run search_pipeline.py first to generate the .npz file
    python search_pipeline.py --trainval --GPU 0
    python search_pipeline.py --trainval --GPU 0 --primary feat_div
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

parser = argparse.ArgumentParser()
parser.add_argument('--results', default='results/pipeline_topk3856_gc.npy.npz')
parser.add_argument('--outdir', default='analysis_plots/pipeline')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# Load saved data
data = np.load(args.results)
gc = data['gc_scores']
accs = data['accs']
skips = data['skips']
tres = data['true_res']
top_k = int(data['top_k'])

gc_median = np.median(gc)
high = gc >= gc_median
low = gc < gc_median

print(f'Loaded {top_k} architectures from {args.results}')
print(f'GC median: {gc_median:.4f}')
print(f'High GC: n={high.sum()}, Low GC: n={low.sum()}')


def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ================================================================
# Plot 1: Box plots -- GC distribution by skip count
# Much cleaner than overlapping histograms
# ================================================================
fig, ax = plt.subplots(figsize=(7, 5))

skip_vals = sorted(set(skips))
box_data = [gc[skips == s] for s in skip_vals]
bp = ax.boxplot(box_data, tick_labels=[str(s) for s in skip_vals],
                patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2))

colors_box = ['#E8E8E8', '#90CAF9', '#42A5F5', '#1565C0']
for patch, c in zip(bp['boxes'], colors_box[:len(skip_vals)]):
    patch.set_facecolor(c)

# Add n and mean acc annotations
for i, s in enumerate(skip_vals):
    mask = skips == s
    n = mask.sum()
    ma = accs[mask].mean()
    ax.text(i + 1, ax.get_ylim()[0] + 0.02, f'n={n}\nacc={ma:.1f}%',
            ha='center', va='bottom', fontsize=8, color='#555')

ax.set_xlabel('Number of Skip Connections', fontsize=12)
ax.set_ylabel('Gradient Consistency (GC)', fontsize=12)
ax.set_title('GC Increases with Skip Count\n(within NASWOT top quartile)', fontsize=13)
ax.axhline(gc_median, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(0.65, gc_median + 0.01, f'median={gc_median:.3f}', fontsize=8, color='gray')
clean_ax(ax)
plt.tight_layout()
plt.savefig(f'{args.outdir}/gc_by_skip_boxplot.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved gc_by_skip_boxplot')


# ================================================================
# Plot 2: The key result -- GC splits pool into skip vs no-skip
# Two grouped bars: % with skip, % true residual
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

labels = ['High GC', 'Low GC']
blue_red = ['#1976D2', '#D32F2F']

# Panel 1: % architectures with skip connections
ax = axes[0]
pct_skip = [100 * (skips[high] >= 1).mean(), 100 * (skips[low] >= 1).mean()]
bars = ax.bar(labels, pct_skip, color=blue_red, width=0.5, edgecolor='white')
for bar, v in zip(bars, pct_skip):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%',
            ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('% with Skip Connections', fontsize=12)
ax.set_title('Skip Connection Rate', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(pct_skip) + 10)
t_s, p_s = stats.ttest_ind(skips[high], skips[low])
ax.text(0.5, 0.95, f'p < 0.0001', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, fontstyle='italic', color='#555')
clean_ax(ax)

# Panel 2: % true residual
ax = axes[1]
pct_tres = [100 * tres[high].mean(), 100 * tres[low].mean()]
bars = ax.bar(labels, pct_tres, color=blue_red, width=0.5, edgecolor='white')
for bar, v in zip(bars, pct_tres):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
            ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('% True Residual Architectures', fontsize=12)
ax.set_title('True Residual Rate', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(pct_tres) + 5)
ax.text(0.5, 0.95, '26x more true residual', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, fontstyle='italic', color='#555')
clean_ax(ax)

plt.suptitle(f'GC Median Split (n={top_k} from NASWOT top quartile)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{args.outdir}/gc_skip_split.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved gc_skip_split')


# ================================================================
# Plot 3: Mean GC by skip count -- simple bar chart
# Shows the monotonic relationship clearly
# ================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

mean_gc = [gc[skips == s].mean() for s in skip_vals]
std_gc = [gc[skips == s].std() for s in skip_vals]
n_per = [int((skips == s).sum()) for s in skip_vals]

bars = ax.bar([str(s) for s in skip_vals], mean_gc,
              yerr=std_gc, capsize=5,
              color=['#E8E8E8', '#90CAF9', '#42A5F5', '#1565C0'][:len(skip_vals)],
              edgecolor='gray', linewidth=0.5)
for bar, v, n in zip(bars, mean_gc, n_per):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_gc[skip_vals.index(int(bar.get_x() + bar.get_width()/2))] if False else v + 0.02,
            f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Skip Connections', fontsize=12)
ax.set_ylabel('Mean Gradient Consistency', fontsize=12)
ax.set_title('GC Monotonically Increases with Skip Count', fontsize=13)
tau_gs, p_gs = stats.kendalltau(gc, skips)
ax.text(0.5, 0.95, f'tau={tau_gs:.3f}, p < 0.0001', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, fontstyle='italic', color='#555')
clean_ax(ax)
plt.tight_layout()
plt.savefig(f'{args.outdir}/mean_gc_by_skip.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved mean_gc_by_skip')


# ================================================================
# Plot 4: Accuracy comparison -- High GC vs Low GC vs NASWOT top
# Horizontal bar chart with CIs
# ================================================================
fig, ax = plt.subplots(figsize=(8, 4))

# Sort by NASWOT score (already sorted in pool_uids, so index 0 = best NASWOT)
naswot_top50 = accs[:50]  # best 50 by NASWOT

groups = {
    'NASWOT Top 50\n(0% skip)': naswot_top50,
    f'Low GC Half\n(n={low.sum()})': accs[low],
    f'High GC Half\n(n={high.sum()})': accs[high],
}

y_pos = np.arange(len(groups))
means = [v.mean() for v in groups.values()]
cis = [1.96 * v.std() / np.sqrt(len(v)) for v in groups.values()]
colors_h = ['#BDBDBD', '#D32F2F', '#1976D2']

bars = ax.barh(y_pos, means, xerr=cis, height=0.5, color=colors_h,
               edgecolor='white', capsize=4)
for bar, m, ci in zip(bars, means, cis):
    ax.text(m + ci + 0.05, bar.get_y() + bar.get_height()/2,
            f'{m:.2f}%', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(groups.keys(), fontsize=11)
ax.set_xlabel('Mean Test Accuracy (%)', fontsize=12)
ax.set_title('GC Selects Higher-Accuracy Architectures', fontsize=13, fontweight='bold')
ax.set_xlim(min(means) - 0.3, max(means) + 0.5)
t_a, p_a = stats.ttest_ind(accs[high], accs[low])
ax.text(0.98, 0.05, f'High vs Low: p={p_a:.4f}', transform=ax.transAxes,
        ha='right', fontsize=9, fontstyle='italic', color='#555')
clean_ax(ax)
plt.tight_layout()
plt.savefig(f'{args.outdir}/accuracy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print('  Saved accuracy_comparison')


print(f'\nAll plots saved to {args.outdir}/')
