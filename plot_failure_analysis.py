"""
plot_failure_analysis.py
========================
Generate presentation-ready plots from failure analysis + attribution results.

Usage
-----
python plot_failure_analysis.py \
    --scores results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy \
    --accs results/full_baseline_accs_nasbench201_cifar10_True.npy \
    --api_loc ../NAS-Bench-201-v1_0-e61699.pth
"""

import argparse
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from collections import Counter

SAVE_DIR = os.path.join('analysis_plots', 'failure_analysis')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument('--scores', required=True, type=str)
parser.add_argument('--accs', required=True, type=str)
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--failure_dir', default='failure_sets', type=str)
args = parser.parse_args()

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
scores_raw = np.load(args.scores)
accs_raw = np.load(args.accs)

valid = (scores_raw != 0) & np.isfinite(scores_raw) & (accs_raw > 0)
valid_uids = np.where(valid)[0]
scores = scores_raw[valid]
accs = accs_raw[valid]

tau_overall, _ = stats.kendalltau(accs, scores)
print(f'Valid architectures: {len(scores)}, overall tau: {tau_overall:.4f}')

# Ranks for inversion analysis
score_ranks = stats.rankdata(scores)
acc_ranks = stats.rankdata(accs)
rank_diff = score_ranks - acc_ranks

overrated_thresh = np.percentile(rank_diff, 95)
underrated_thresh = np.percentile(rank_diff, 5)
overrated = rank_diff >= overrated_thresh
underrated = rank_diff <= underrated_thresh
middle = ~overrated & ~underrated


# =========================================================================
# PLOT 1: Score vs Accuracy scatter with overrated/underrated highlighted
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(accs[middle], scores[middle], s=3, alpha=0.15, c='#888888',
           label='Correctly ranked', zorder=1, rasterized=True)
ax.scatter(accs[overrated], scores[overrated], s=8, alpha=0.5, c='#D32F2F',
           label=f'Overrated (n={overrated.sum()})', zorder=2)
ax.scatter(accs[underrated], scores[underrated], s=8, alpha=0.5, c='#1976D2',
           label=f'Underrated (n={underrated.sum()})', zorder=2)

ax.set_xlabel('Validation Accuracy (%)')
ax.set_ylabel('NASWOT Score (log det K)')
ax.set_title(f'NASWOT Failures on NAS-Bench-201 CIFAR-10\n'
             f'Overall $\\tau$ = {tau_overall:.3f}')
ax.legend(loc='lower right', markerscale=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '1_score_vs_accuracy_failures.png'), dpi=200)
plt.savefig(os.path.join(SAVE_DIR, '1_score_vs_accuracy_failures.pdf'))
plt.close()
print('Saved: 1_score_vs_accuracy_failures')


# =========================================================================
# PLOT 2: Per-band Kendall tau (where NASWOT breaks down)
# =========================================================================
bands = [
    ('Bottom 25%', 0, 25),
    ('25-50%', 25, 50),
    ('50-75%', 50, 75),
    ('Top 25%', 75, 100),
]
band_taus = []
band_labels = []
band_ns = []
for label, lo, hi in bands:
    lo_val = np.percentile(accs, lo)
    hi_val = np.percentile(accs, hi)
    if lo == 0:
        mask = accs <= hi_val
    elif hi == 100:
        mask = accs > lo_val
    else:
        mask = (accs > lo_val) & (accs <= hi_val)
    if mask.sum() > 10:
        t, _ = stats.kendalltau(accs[mask], scores[mask])
        band_taus.append(t)
        band_labels.append(f'{label}\n({lo_val:.0f}-{hi_val:.0f}%)\nn={mask.sum()}')
        band_ns.append(mask.sum())

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
bars = ax.bar(range(len(band_taus)), band_taus, color=colors, edgecolor='black',
              linewidth=0.5)
for i, (b, t) in enumerate(zip(bars, band_taus)):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f'{t:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(band_labels)))
ax.set_xticklabels(band_labels, fontsize=9)
ax.set_ylabel('Kendall $\\tau$')
ax.set_title('NASWOT Ranking Quality by Accuracy Band')
ax.set_ylim(0, max(band_taus) * 1.2)
ax.axhline(y=tau_overall, color='black', linestyle='--', alpha=0.5,
           label=f'Overall $\\tau$ = {tau_overall:.3f}')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '2_tau_by_accuracy_band.png'), dpi=200)
plt.savefig(os.path.join(SAVE_DIR, '2_tau_by_accuracy_band.pdf'))
plt.close()
print('Saved: 2_tau_by_accuracy_band')


# =========================================================================
# PLOT 3: Score distributions by accuracy quartile (overlap visualization)
# =========================================================================
quartile_thresholds = np.percentile(accs, [25, 50, 75])
q_masks = [
    accs <= quartile_thresholds[0],
    (accs > quartile_thresholds[0]) & (accs <= quartile_thresholds[1]),
    (accs > quartile_thresholds[1]) & (accs <= quartile_thresholds[2]),
    accs > quartile_thresholds[2],
]
q_labels = [
    f'Q1 (acc $\\leq$ {quartile_thresholds[0]:.0f}%)',
    f'Q2 ({quartile_thresholds[0]:.0f}-{quartile_thresholds[1]:.0f}%)',
    f'Q3 ({quartile_thresholds[1]:.0f}-{quartile_thresholds[2]:.0f}%)',
    f'Q4 (acc > {quartile_thresholds[2]:.0f}%)',
]
q_colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50']

fig, ax = plt.subplots(figsize=(7, 4))
for mask, label, color in zip(q_masks, q_labels, q_colors):
    ax.hist(scores[mask], bins=60, alpha=0.45, label=label, color=color,
            density=True, edgecolor='none')

ax.set_xlabel('NASWOT Score (log det K)')
ax.set_ylabel('Density')
ax.set_title('Score Distribution by Accuracy Quartile\n'
             'Q2-Q4 overlap explains low $\\tau$ in upper bands')
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '3_score_distribution_by_quartile.png'), dpi=200)
plt.savefig(os.path.join(SAVE_DIR, '3_score_distribution_by_quartile.pdf'))
plt.close()
print('Saved: 3_score_distribution_by_quartile')


# =========================================================================
# PLOT 4: Operation distribution -- overrated vs underrated vs all
# =========================================================================
print('Loading NAS-Bench-201 API for architecture analysis...')
from nas_201_api import NASBench201API as API
api = API(args.api_loc, verbose=False)

OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
OP_LABELS = ['none', 'skip', 'conv1x1', 'conv3x3', 'pool3x3']


def parse_arch_str(arch_str):
    ops = []
    for part in arch_str.split('+'):
        part = part.strip('|')
        for token in part.split('|'):
            token = token.strip()
            if '~' in token:
                ops.append(token.split('~')[0])
    return ops


def op_dist(uid_set):
    counts = Counter()
    for uid in uid_set:
        config = api.get_net_config(int(uid), 'cifar10-valid')
        ops = parse_arch_str(config['arch_str'])
        for op in ops:
            counts[op] += 1
    total = len(uid_set) * 6
    return {op: counts.get(op, 0) / total for op in OPS}


overrated_uids = valid_uids[overrated]
underrated_uids = valid_uids[underrated]

print('Computing op distributions...')
over_dist = op_dist(overrated_uids)
under_dist = op_dist(underrated_uids)
all_dist = op_dist(valid_uids)

x = np.arange(len(OPS))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar(x - width, [over_dist[op] for op in OPS], width,
               label='Overrated', color='#D32F2F', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, [all_dist[op] for op in OPS], width,
               label='All', color='#9E9E9E', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, [under_dist[op] for op in OPS], width,
               label='Underrated', color='#1976D2', edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(OP_LABELS)
ax.set_ylabel('Fraction of Op Slots')
ax.set_title('Operation Distribution: Overrated vs Underrated Architectures')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate the biggest differences
for i, op in enumerate(OPS):
    if op == 'nor_conv_1x1':
        ax.annotate(f'{over_dist[op]:.0%}', xy=(i - width, over_dist[op]),
                     xytext=(i - width - 0.15, over_dist[op] + 0.04),
                     fontsize=9, fontweight='bold', color='#D32F2F')
    elif op == 'skip_connect':
        ax.annotate(f'{under_dist[op]:.0%}', xy=(i + width, under_dist[op]),
                     xytext=(i + width - 0.1, under_dist[op] + 0.04),
                     fontsize=9, fontweight='bold', color='#1976D2')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '4_op_distribution_overrated_vs_underrated.png'), dpi=200)
plt.savefig(os.path.join(SAVE_DIR, '4_op_distribution_overrated_vs_underrated.pdf'))
plt.close()
print('Saved: 4_op_distribution_overrated_vs_underrated')


# =========================================================================
# PLOT 5: Rank diff histogram
# =========================================================================
fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(rank_diff[middle], bins=80, color='#888888', alpha=0.6, density=True,
        label='Correctly ranked', edgecolor='none')
ax.hist(rank_diff[overrated], bins=30, color='#D32F2F', alpha=0.7, density=True,
        label='Overrated (top 5% rank diff)', edgecolor='none')
ax.hist(rank_diff[underrated], bins=30, color='#1976D2', alpha=0.7, density=True,
        label='Underrated (bottom 5% rank diff)', edgecolor='none')

ax.axvline(x=0, color='black', linewidth=1, linestyle='-')
ax.set_xlabel('Score Rank $-$ Accuracy Rank')
ax.set_ylabel('Density')
ax.set_title('Rank Inversion Distribution\n'
             'Positive = NASWOT overrates, Negative = underrates')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '5_rank_inversion_histogram.png'), dpi=200)
plt.savefig(os.path.join(SAVE_DIR, '5_rank_inversion_histogram.pdf'))
plt.close()
print('Saved: 5_rank_inversion_histogram')


# =========================================================================
# PLOT 6: Complexity confound -- score vs #conv3x3 and #conv3x3 vs accuracy
# =========================================================================
print('Computing per-architecture op counts...')
all_conv3 = np.zeros(len(valid_uids))
all_skip = np.zeros(len(valid_uids))
all_none_count = np.zeros(len(valid_uids))
for i, uid in enumerate(valid_uids):
    config = api.get_net_config(int(uid), 'cifar10-valid')
    ops = parse_arch_str(config['arch_str'])
    all_conv3[i] = ops.count('nor_conv_3x3')
    all_skip[i] = ops.count('skip_connect')
    all_none_count[i] = ops.count('none')
    if (i + 1) % 3000 == 0:
        print(f'  {i+1}/{len(valid_uids)}...')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: score vs #conv3x3
tau_sc, _ = stats.kendalltau(scores, all_conv3)
jitter = np.random.normal(0, 0.12, len(all_conv3))
axes[0].scatter(all_conv3 + jitter, scores, s=2, alpha=0.1, c='#555555',
                rasterized=True)
# Mean per bin
for n3 in range(7):
    mask = all_conv3 == n3
    if mask.sum() > 0:
        axes[0].plot(n3, scores[mask].mean(), 'o', color='#D32F2F',
                     markersize=8, zorder=5)
axes[0].set_xlabel('Number of conv3x3 ops')
axes[0].set_ylabel('NASWOT Score')
axes[0].set_title(f'Score vs #conv3x3\n$\\tau$ = {tau_sc:.3f}')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Right: #conv3x3 vs accuracy
tau_ca, _ = stats.kendalltau(all_conv3, accs)
axes[1].scatter(all_conv3 + jitter, accs, s=2, alpha=0.1, c='#555555',
                rasterized=True)
for n3 in range(7):
    mask = all_conv3 == n3
    if mask.sum() > 0:
        axes[1].plot(n3, accs[mask].mean(), 'o', color='#1976D2',
                     markersize=8, zorder=5)
axes[1].set_xlabel('Number of conv3x3 ops')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].set_title(f'Accuracy vs #conv3x3\n$\\tau$ = {tau_ca:.3f}')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.suptitle('Complexity Confound: NASWOT partly tracks op count', y=1.02,
             fontsize=BIGGER_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '6_complexity_confound.png'), dpi=200,
            bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '6_complexity_confound.pdf'),
            bbox_inches='tight')
plt.close()
print('Saved: 6_complexity_confound')


# =========================================================================
# PLOT 7: Summary figure -- key numbers at a glance
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

summary_text = f"""NASWOT Failure Analysis Summary
NAS-Bench-201 / CIFAR-10 / 15,424 valid architectures (201 singular)

Overall Kendall tau:  {tau_overall:.4f}

Tau by accuracy band:
    Bottom 25% (acc < {quartile_thresholds[0]:.0f}%):   tau = {band_taus[0]:.3f}   (separates bad from decent)
    25-50% ({quartile_thresholds[0]:.0f}-{quartile_thresholds[1]:.0f}%):          tau = {band_taus[1]:.3f}   (weak discrimination)
    50-75% ({quartile_thresholds[1]:.0f}-{quartile_thresholds[2]:.0f}%):          tau = {band_taus[2]:.3f}   (nearly random)
    Top 25% (acc > {quartile_thresholds[2]:.0f}%):          tau = {band_taus[3]:.3f}   (nearly random)

Failure diagnosis:
    772 overrated architectures  (score rank >> accuracy rank)
    772 underrated architectures (score rank << accuracy rank)

Structural root cause:
    Overrated:   43% conv1x1,  7% skip   (inflated score from cheap activations)
    Underrated:   7% conv1x1, 28% skip   (good trainability, low score)

    tau(score, #conv3x3) = {tau_sc:.3f}   -> score partly tracks complexity
    tau(#conv3x3, acc)   = {tau_ca:.3f}   -> complexity does predict accuracy

Implication:
    NASWOT rewards activation count, not architecture quality.
    Skip connections improve training but contribute nothing to log det K.
    The top-50% ranking (where NAS matters) is nearly random."""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '7_summary.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '7_summary.pdf'), bbox_inches='tight')
plt.close()
print('Saved: 7_summary')

print(f'\nAll plots saved to {SAVE_DIR}/')
