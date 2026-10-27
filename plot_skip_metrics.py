"""
plot_skip_metrics.py
====================
Visualize which metric best separates overrated vs underrated architectures.
Loads fewstep results + failure sets and creates comparison plots.

Usage:
    python plot_skip_metrics.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SAVE_DIR = os.path.join('analysis_plots', 'skip_metrics')
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================
prefix = 'results/fewstep_200_lr0.1'
accs = np.load(f'{prefix}_accs.npy')
n_skips = np.load(f'{prefix}_nskips.npy')
true_res = np.load(f'{prefix}_true_res.npy')

metrics = {
    'NASWOT logdet':       np.load(f'{prefix}_logdet_init.npy'),
    'Grad consistency':    np.load(f'{prefix}_grad_consistency.npy'),
    'Neg val loss (20)':   np.load(f'{prefix}_neg_val_loss_20.npy'),
    'Val acc (20 steps)':  np.load(f'{prefix}_val_acc_20.npy'),
    'NASWOT delta (20)':   np.load(f'{prefix}_naswot_delta_20.npy'),
    'Neg val loss (10)':   np.load(f'{prefix}_neg_val_loss_10.npy'),
}

# Load overrated/underrated sets
overrated_uids = set(np.load('failure_sets/overrated_uids.npy').tolist())
underrated_uids = set(np.load('failure_sets/underrated_uids.npy').tolist())

valid = (accs > 0) & np.all([np.isfinite(m) for m in metrics.values()], axis=0)

# Masks for overrated/underrated within first 200 UIDs
over_mask = np.array([i in overrated_uids for i in range(200)]) & valid
under_mask = np.array([i in underrated_uids for i in range(200)]) & valid
skip_mask = (n_skips >= 1) & valid
noskip_mask = (n_skips == 0) & valid

print(f'Valid: {valid.sum()}, Overrated: {over_mask.sum()}, Underrated: {under_mask.sum()}')
print(f'Has skip: {skip_mask.sum()}, No skip: {noskip_mask.sum()}')


# ============================================================
# PLOT 1: Overrated vs Underrated distributions for each metric
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, (name, vals) in zip(axes, metrics.items()):
    over_vals = vals[over_mask]
    under_vals = vals[under_mask]

    over_vals = over_vals[np.isfinite(over_vals)]
    under_vals = under_vals[np.isfinite(under_vals)]

    if len(over_vals) == 0 or len(under_vals) == 0:
        ax.set_title(f'{name}\n(insufficient data)')
        continue

    # Cohen's d (effect size)
    pooled_std = np.sqrt((over_vals.std()**2 + under_vals.std()**2) / 2)
    cohens_d = (under_vals.mean() - over_vals.mean()) / (pooled_std + 1e-10)

    # t-test
    t_stat, p_val = stats.ttest_ind(under_vals, over_vals)

    ax.hist(over_vals, bins=15, alpha=0.6, color='#D32F2F', label='Overrated', density=True)
    ax.hist(under_vals, bins=15, alpha=0.6, color='#1976D2', label='Underrated', density=True)
    ax.axvline(over_vals.mean(), color='#D32F2F', linestyle='--', linewidth=2)
    ax.axvline(under_vals.mean(), color='#1976D2', linestyle='--', linewidth=2)

    # Direction check
    correct = (under_vals.mean() > over_vals.mean())
    direction = "CORRECT" if correct else "WRONG"

    ax.set_title(f'{name}\nd={cohens_d:.2f}, p={p_val:.3f} ({direction})', fontsize=11)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Overrated vs Underrated: Which metric separates them best?\n'
             '(Cohen\'s d = effect size; higher = better separation)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1_overrated_vs_underrated_distributions.png'),
            dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '1_overrated_vs_underrated_distributions.pdf'),
            bbox_inches='tight')
plt.close()
print(f'Plot 1 saved')


# ============================================================
# PLOT 2: Effect size comparison (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

names = []
effect_sizes = []
p_values = []
directions = []

for name, vals in metrics.items():
    over_vals = vals[over_mask][np.isfinite(vals[over_mask])]
    under_vals = vals[under_mask][np.isfinite(vals[under_mask])]
    if len(over_vals) < 2 or len(under_vals) < 2:
        continue

    pooled_std = np.sqrt((over_vals.std()**2 + under_vals.std()**2) / 2)
    d = (under_vals.mean() - over_vals.mean()) / (pooled_std + 1e-10)
    _, p = stats.ttest_ind(under_vals, over_vals)

    names.append(name)
    effect_sizes.append(d)
    p_values.append(p)
    directions.append('correct' if d > 0 else 'wrong')

colors = ['#4CAF50' if d == 'correct' else '#D32F2F' for d in directions]
bars = ax.barh(range(len(names)), effect_sizes, color=colors, edgecolor='black', linewidth=0.5)

# Add p-value labels
for i, (es, p) in enumerate(zip(effect_sizes, p_values)):
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(es + 0.02 if es >= 0 else es - 0.02, i,
            f' d={es:.2f} {sig}', va='center',
            ha='left' if es >= 0 else 'right', fontsize=10)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel("Cohen's d (effect size)\nPositive = underrated scores HIGHER (correct direction)", fontsize=11)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('Which metric best separates overrated from underrated architectures?',
             fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '2_effect_size_comparison.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '2_effect_size_comparison.pdf'), bbox_inches='tight')
plt.close()
print(f'Plot 2 saved')


# ============================================================
# PLOT 3: Scatter plots - metric vs accuracy, colored by skip count
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

cmap = plt.cm.viridis
skip_colors = {0: '#D32F2F', 1: '#FF9800', 2: '#4CAF50', 3: '#1976D2', 4: '#9C27B0', 5: '#795548'}

for ax, (name, vals) in zip(axes, metrics.items()):
    v = valid & np.isfinite(vals)
    for ns in sorted(set(n_skips[v])):
        mask = v & (n_skips == ns)
        if mask.sum() == 0:
            continue
        ax.scatter(vals[mask], accs[mask], alpha=0.5, s=20,
                   color=skip_colors.get(ns, '#999'),
                   label=f'{ns} skips (n={mask.sum()})')

    # Tau
    tau, _ = stats.kendalltau(accs[v], vals[v])
    ax.set_xlabel(name)
    ax.set_ylabel('Ground truth accuracy (%)')
    ax.set_title(f'{name}\ntau={tau:.3f}', fontsize=11)
    ax.legend(fontsize=7, markerscale=1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Score vs Accuracy (colored by skip count)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '3_scatter_by_skip_count.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '3_scatter_by_skip_count.pdf'), bbox_inches='tight')
plt.close()
print(f'Plot 3 saved')


# ============================================================
# PLOT 4: Skip vs NoSkip mean scores (grouped bar chart)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Select the three candidate metrics
candidates = ['Grad consistency', 'Neg val loss (20)', 'NASWOT delta (20)']

for ax, name in zip(axes, candidates):
    vals = metrics[name]
    groups = {
        'No skip\n(n_skip=0)': noskip_mask,
        'Has skip\n(n_skip>=1)': skip_mask,
        'True\nresidual': (true_res == 1) & valid,
        'Overrated': over_mask,
        'Underrated': under_mask,
    }

    group_names = list(groups.keys())
    group_means = []
    group_stds = []
    colors_list = ['#D32F2F', '#4CAF50', '#1976D2', '#FF9800', '#9C27B0']

    for gname, gmask in groups.items():
        v = vals[gmask]
        v = v[np.isfinite(v)]
        group_means.append(v.mean() if len(v) > 0 else 0)
        group_stds.append(v.std() / np.sqrt(len(v)) if len(v) > 1 else 0)

    bars = ax.bar(range(len(group_names)), group_means, yerr=group_stds,
                  color=colors_list, edgecolor='black', linewidth=0.5,
                  capsize=4, alpha=0.8)
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, fontsize=9)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on bars
    for bar, mean_val in zip(bars, group_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Three candidate metrics: Mean score by architecture group\n'
             '(Underrated should score HIGHER than Overrated)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '4_candidate_metrics_by_group.png'),
            dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '4_candidate_metrics_by_group.pdf'),
            bbox_inches='tight')
plt.close()
print(f'Plot 4 saved')


# ============================================================
# PLOT 5: NASWOT logdet direction is WRONG (the key insight)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: NASWOT logdet (wrong direction)
ax = axes[0]
over_ld = metrics['NASWOT logdet'][over_mask]
under_ld = metrics['NASWOT logdet'][under_mask]
ax.hist(over_ld[np.isfinite(over_ld)], bins=12, alpha=0.6, color='#D32F2F',
        label=f'Overrated (acc={accs[over_mask].mean():.1f}%)', density=True)
ax.hist(under_ld[np.isfinite(under_ld)], bins=12, alpha=0.6, color='#1976D2',
        label=f'Underrated (acc={accs[under_mask].mean():.1f}%)', density=True)
ax.axvline(over_ld[np.isfinite(over_ld)].mean(), color='#D32F2F', linestyle='--', linewidth=2)
ax.axvline(under_ld[np.isfinite(under_ld)].mean(), color='#1976D2', linestyle='--', linewidth=2)
ax.set_xlabel('NASWOT logdet score')
ax.set_title('NASWOT: Overrated scores HIGHER (WRONG)', fontsize=12, color='#D32F2F')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: Best new metric (correct direction)
ax = axes[1]
# Find the metric with the best Cohen's d
best_name, best_d = None, -999
for name, vals in metrics.items():
    if name == 'NASWOT logdet':
        continue
    ov = vals[over_mask][np.isfinite(vals[over_mask])]
    un = vals[under_mask][np.isfinite(vals[under_mask])]
    if len(ov) < 2 or len(un) < 2:
        continue
    ps = np.sqrt((ov.std()**2 + un.std()**2) / 2)
    d = (un.mean() - ov.mean()) / (ps + 1e-10)
    if d > best_d:
        best_d = d
        best_name = name

best_vals = metrics[best_name]
over_bv = best_vals[over_mask]
under_bv = best_vals[under_mask]
ax.hist(over_bv[np.isfinite(over_bv)], bins=12, alpha=0.6, color='#D32F2F',
        label=f'Overrated (acc={accs[over_mask].mean():.1f}%)', density=True)
ax.hist(under_bv[np.isfinite(under_bv)], bins=12, alpha=0.6, color='#1976D2',
        label=f'Underrated (acc={accs[under_mask].mean():.1f}%)', density=True)
ax.axvline(over_bv[np.isfinite(over_bv)].mean(), color='#D32F2F', linestyle='--', linewidth=2)
ax.axvline(under_bv[np.isfinite(under_bv)].mean(), color='#1976D2', linestyle='--', linewidth=2)
ax.set_xlabel(best_name)
ax.set_title(f'{best_name}: Underrated scores HIGHER (CORRECT)\nCohen\'s d = {best_d:.2f}',
             fontsize=12, color='#1976D2')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('NASWOT gets it backwards. The new metric gets it right.',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '5_naswot_wrong_vs_new_correct.png'),
            dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '5_naswot_wrong_vs_new_correct.pdf'),
            bbox_inches='tight')
plt.close()
print(f'Plot 5 saved (best metric: {best_name}, d={best_d:.2f})')


# ============================================================
# Print summary: which metric is actually best?
# ============================================================
print(f'\n{"="*60}')
print('METRIC COMPARISON SUMMARY')
print(f'{"="*60}')
print(f'{"Metric":<25s} {"Cohen d":>8s} {"p-value":>10s} {"Direction":>10s} {"Cost":>10s}')
print('-' * 65)

for name, vals in metrics.items():
    ov = vals[over_mask][np.isfinite(vals[over_mask])]
    un = vals[under_mask][np.isfinite(vals[under_mask])]
    if len(ov) < 2 or len(un) < 2:
        continue
    ps = np.sqrt((ov.std()**2 + un.std()**2) / 2)
    d = (un.mean() - ov.mean()) / (ps + 1e-10)
    _, p = stats.ttest_ind(un, ov)
    direction = 'CORRECT' if d > 0 else 'WRONG'
    cost = 'zero-cost' if name in ['NASWOT logdet', 'Grad consistency'] else '20 steps'
    print(f'{name:<25s} {d:>+8.2f} {p:>10.4f} {direction:>10s} {cost:>10s}')

print(f'\nBest metric by effect size: {best_name} (d={best_d:.2f})')
print(f'\nAll plots saved to {SAVE_DIR}/')
