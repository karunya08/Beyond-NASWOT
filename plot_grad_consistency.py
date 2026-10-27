"""
plot_grad_consistency.py
========================
Plots for gradient consistency results.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

SAVE_DIR = os.path.join('analysis_plots', 'grad_consistency')
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data
prefix = 'results/grad_consistency_200'
accs = np.load(f'{prefix}_accs.npy')
gc = np.load(f'{prefix}_gc.npy')
logdet = np.load(f'{prefix}_logdet.npy')
n_skips = np.load(f'{prefix}_nskips.npy')
true_res = np.load(f'{prefix}_true_res.npy')

# Load neg val loss from fewstep results
prefix2 = 'results/fewstep_200_lr0.1'
neg_val_loss = np.load(f'{prefix2}_neg_val_loss_20.npy')

valid = (accs > 0) & np.isfinite(gc) & np.isfinite(logdet) & np.isfinite(neg_val_loss)

# Load overrated/underrated
overrated_uids = set(np.load('failure_sets/overrated_uids.npy').tolist())
underrated_uids = set(np.load('failure_sets/underrated_uids.npy').tolist())
over_mask = np.array([i in overrated_uids for i in range(200)]) & valid
under_mask = np.array([i in underrated_uids for i in range(200)]) & valid


# ============================================================
# Helper: plot overrated vs underrated histogram for one metric
# ============================================================
def plot_ov_un_hist(ax, vals_over, vals_under, xlabel, title_label, cost_label, acc_ov, acc_un):
    ps = np.sqrt((vals_over.std()**2 + vals_under.std()**2) / 2)
    d = (vals_under.mean() - vals_over.mean()) / (ps + 1e-10)
    _, p = stats.ttest_ind(vals_under, vals_over)
    correct = vals_under.mean() > vals_over.mean()
    direction = 'CORRECT' if correct else 'WRONG direction'
    color = '#1976D2' if correct else '#D32F2F'

    ax.hist(vals_over, bins=10, alpha=0.6, color='#D32F2F',
            label=f'Overrated (acc={acc_ov:.1f}%)', density=True)
    ax.hist(vals_under, bins=10, alpha=0.6, color='#1976D2',
            label=f'Underrated (acc={acc_un:.1f}%)', density=True)
    ax.axvline(vals_over.mean(), color='#D32F2F', linestyle='--', linewidth=2)
    ax.axvline(vals_under.mean(), color='#1976D2', linestyle='--', linewidth=2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f'{title_label} ({cost_label})\nd={d:.2f}, p={p:.4f} -- {direction}',
                 fontsize=11, color=color)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

acc_ov = accs[over_mask].mean()
acc_un = accs[under_mask].mean()

# ============================================================
# PLOT 1a: NASWOT vs Grad Consistency (2 panels)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_ov_un_hist(axes[0], logdet[over_mask], logdet[under_mask],
                'NASWOT logdet score', 'NASWOT logdet', 'zero-cost', acc_ov, acc_un)
plot_ov_un_hist(axes[1], gc[over_mask], gc[under_mask],
                'Gradient Consistency (cosine sim)', 'Grad Consistency', 'zero-cost', acc_ov, acc_un)

plt.suptitle('NASWOT gets it backwards. Gradient consistency gets it right.',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'naswot_vs_gc.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'naswot_vs_gc.pdf'), bbox_inches='tight')
plt.close()
print('Plot 1a saved: naswot_vs_gc (2 panels)')

# ============================================================
# PLOT 1b: NASWOT vs Grad Consistency vs Neg Val Loss (3 panels)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_ov_un_hist(axes[0], logdet[over_mask], logdet[under_mask],
                'NASWOT logdet score', 'NASWOT logdet', 'zero-cost', acc_ov, acc_un)
plot_ov_un_hist(axes[1], gc[over_mask], gc[under_mask],
                'Gradient Consistency (cosine sim)', 'Grad Consistency', 'zero-cost', acc_ov, acc_un)
plot_ov_un_hist(axes[2], neg_val_loss[over_mask], neg_val_loss[under_mask],
                'Neg Val Loss (after 20 SGD steps)', 'Neg Val Loss', '20 steps', acc_ov, acc_un)

plt.suptitle('Overrated vs Underrated: NASWOT fails, both new metrics succeed',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'naswot_vs_gc_vs_nvl.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'naswot_vs_gc_vs_nvl.pdf'), bbox_inches='tight')
plt.close()
print('Plot 1b saved: naswot_vs_gc_vs_nvl (3 panels)')


# ============================================================
# PLOT 2: Grad consistency by skip count (bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

skip_counts = sorted(set(n_skips[valid]))
means = []
stds = []
ns = []
for s in skip_counts:
    mask = (n_skips == s) & valid
    vals = gc[mask]
    means.append(vals.mean())
    stds.append(vals.std() / np.sqrt(len(vals)))
    ns.append(mask.sum())

colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(skip_counts)))
bars = ax.bar(range(len(skip_counts)), means, yerr=stds, color=colors,
              edgecolor='black', linewidth=0.5, capsize=5)

for bar, m, n in zip(bars, means, ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{m:.3f}\n(n={n})', ha='center', va='bottom', fontsize=10)

ax.set_xticks(range(len(skip_counts)))
ax.set_xticklabels([f'{s} skips' for s in skip_counts], fontsize=11)
ax.set_ylabel('Mean Gradient Consistency', fontsize=12)
ax.set_title('Gradient Consistency increases with skip connections',
             fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'gc_by_skip_count.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'gc_by_skip_count.pdf'), bbox_inches='tight')
plt.close()
print('Plot 2 saved: gc_by_skip_count')


# ============================================================
# PLOT 3: Scatter - gc vs accuracy, colored by skip count
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

skip_colors = {0: '#D32F2F', 1: '#FF9800', 2: '#4CAF50', 3: '#1976D2', 4: '#9C27B0', 5: '#795548'}

# Left: NASWOT vs accuracy
ax = axes[0]
for ns in sorted(set(n_skips[valid])):
    mask = (n_skips == ns) & valid
    ax.scatter(logdet[mask], accs[mask], alpha=0.5, s=25,
               color=skip_colors.get(ns, '#999'), label=f'{ns} skips')
# Mark overrated/underrated
ax.scatter(logdet[over_mask], accs[over_mask], s=80, facecolors='none',
           edgecolors='red', linewidths=2, label='Overrated', zorder=5)
ax.scatter(logdet[under_mask], accs[under_mask], s=80, facecolors='none',
           edgecolors='blue', linewidths=2, label='Underrated', zorder=5)
tau, _ = stats.kendalltau(accs[valid], logdet[valid])
ax.set_xlabel('NASWOT logdet', fontsize=11)
ax.set_ylabel('Ground truth accuracy (%)', fontsize=11)
ax.set_title(f'NASWOT logdet vs Accuracy (tau={tau:.3f})', fontsize=12)
ax.legend(fontsize=8, markerscale=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: GC vs accuracy
ax = axes[1]
for ns in sorted(set(n_skips[valid])):
    mask = (n_skips == ns) & valid
    ax.scatter(gc[mask], accs[mask], alpha=0.5, s=25,
               color=skip_colors.get(ns, '#999'), label=f'{ns} skips')
ax.scatter(gc[over_mask], accs[over_mask], s=80, facecolors='none',
           edgecolors='red', linewidths=2, label='Overrated', zorder=5)
ax.scatter(gc[under_mask], accs[under_mask], s=80, facecolors='none',
           edgecolors='blue', linewidths=2, label='Underrated', zorder=5)
tau_gc, _ = stats.kendalltau(accs[valid], gc[valid])
ax.set_xlabel('Gradient Consistency', fontsize=11)
ax.set_ylabel('Ground truth accuracy (%)', fontsize=11)
ax.set_title(f'Grad Consistency vs Accuracy (tau={tau_gc:.3f})', fontsize=12)
ax.legend(fontsize=8, markerscale=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('NASWOT vs Gradient Consistency: skip architectures highlighted',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'scatter_naswot_vs_gc.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, 'scatter_naswot_vs_gc.pdf'), bbox_inches='tight')
plt.close()
print('Plot 3 saved: scatter_naswot_vs_gc')

print(f'\nAll plots saved to {SAVE_DIR}/')
