"""
analyze_cka_systems.py
======================
Load pre-computed scores from score_cka_systems.py and generate
analysis tables + plots.

Usage:
    # After running: python score_cka_systems.py --trainval --GPU 0 ...
    python analyze_cka_systems.py --dataset cifar10
    python analyze_cka_systems.py --dataset cifar100
    python analyze_cka_systems.py --dataset cifar10 --no_plots
"""

import argparse
import os
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser(description='Analyze CKA + Systems scoring results')
parser.add_argument('--save_loc', default='results', type=str)
parser.add_argument('--save_string', default='v2', type=str)
parser.add_argument('--nasspace', default='nasbench201', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--trainval', default=True, type=bool)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Alpha used in score_cka_systems.py (must match)')
parser.add_argument('--no_plots', action='store_true')

args = parser.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD PRE-COMPUTED RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

prefix = f'{args.save_loc}/{args.save_string}'
suffix = f'{args.nasspace}_{args.dataset}_{args.trainval}_{args.seed}'
hw_suffix = f'{args.nasspace}_{args.dataset}_{args.trainval}'

try:
    accs = np.load(f'{prefix}_accs_{hw_suffix}.npy')
except FileNotFoundError:
    print(f'ERROR: Results not found. Run score_cka_systems.py first:')
    print(f'  python score_cka_systems.py --trainval --GPU 0 --dataset {args.dataset} ...')
    exit(1)

scores_baseline = np.load(f'{prefix}_hook_logdet_{suffix}.npy')
scores_cka = np.load(f'{prefix}_layer_cka_weighted_{suffix}.npy')
hw_params = np.load(f'{prefix}_hw_params_{hw_suffix}.npy')
hw_latency = np.load(f'{prefix}_hw_latency_{hw_suffix}.npy')
hw_memory = np.load(f'{prefix}_hw_memory_{hw_suffix}.npy')

alpha = args.alpha
SYSTEMS = [f'sys_params_{alpha}', f'sys_latency_{alpha}',
           f'sys_memory_{alpha}']
scores_sys = {}
for m in SYSTEMS:
    fpath = f'{prefix}_{m}_{suffix}.npy'
    if not os.path.exists(fpath):
        # Try legacy naming (e.g. sys_params_05 instead of sys_params_0.5)
        legacy = m.replace(f'_{alpha}', f'_{str(alpha).replace(".", "")}')
        fpath = f'{prefix}_{legacy}_{suffix}.npy'
    scores_sys[m] = np.load(fpath)

N = len(accs)
print(f'Loaded {N} architectures for {args.dataset}')

# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

valid = ~np.isnan(scores_baseline) & (scores_baseline != 0) & \
        ~np.isnan(scores_cka) & (scores_cka != 0) & (accs > 0)
a = accs[valid]
sb = scores_baseline[valid]
sc = scores_cka[valid]

tau_b, _ = stats.kendalltau(a, sb)
tau_c, _ = stats.kendalltau(a, sc)
rho_b, _ = stats.spearmanr(a, sb)
rho_c, _ = stats.spearmanr(a, sc)

print(f'\n{"="*70}')
print(f'  RESULTS -- {args.dataset} ({valid.sum()} architectures)')
print(f'{"="*70}')

# Correlation metrics
print(f'\n  -- Correlation with True Accuracy --')
print(f'  {"Metric":>20s}  {"Baseline":>10s}  {"CKA-weighted":>12s}  {"Delta":>8s}')
print(f'  {"-"*55}')
print(f'  {"Kendall tau":>20s}  {tau_b:>+10.4f}  {tau_c:>+12.4f}  {tau_c-tau_b:>+8.4f}')
print(f'  {"Spearman rho":>20s}  {rho_b:>+10.4f}  {rho_c:>+12.4f}  {rho_c-rho_b:>+8.4f}')

# Top-K accuracy
print(f'\n  -- Top-K Architecture Selection --')
print(f'  {"K":>5s}  {"Baseline":>10s}  {"CKA-weighted":>12s}  {"Delta":>8s}  {"Oracle":>8s}')
print(f'  {"-"*50}')
for k in [1, 5, 10, 20, 50]:
    if k > valid.sum():
        break
    top_b = a[np.argsort(sb)[-k:]].mean()
    top_c = a[np.argsort(sc)[-k:]].mean()
    top_o = a[np.argsort(a)[-k:]].mean()
    print(f'  {k:>5d}  {top_b:>9.2f}%  {top_c:>11.2f}%  {top_c-top_b:>+7.2f}%  {top_o:>7.2f}%')

# Systems analysis
print(f'\n  -- Systems Analysis: Which architecture does each method pick? --')
print(f'  {"Method":>25s}  {"Accuracy":>8s}  {"Params":>8s}  {"Latency":>10s}  {"Memory":>10s}')
print(f'  {"-"*65}')

all_methods = {
    'hook_logdet (baseline)': scores_baseline,
    'CKA-weighted (ours)': scores_cka,
}
for m in SYSTEMS:
    all_methods[m] = scores_sys[m]

for name, s in all_methods.items():
    sv = s[valid]
    if (sv != 0).sum() == 0:
        continue
    best_idx_local = np.nanargmax(np.where(sv != 0, sv, -np.inf))
    valid_indices = np.where(valid)[0]
    best_idx = valid_indices[best_idx_local]
    print(f'  {name:>25s}  {accs[best_idx]:>7.2f}%  {hw_params[best_idx]/1e3:>7.0f}K'
          f'  {hw_latency[best_idx]*1e3:>8.1f}ms  {hw_memory[best_idx]/1e6:>8.0f}MB')

best_acc_idx = np.argmax(accs)
print(f'  {"(oracle best acc)":>25s}  {accs[best_acc_idx]:>7.2f}%  {hw_params[best_acc_idx]/1e3:>7.0f}K'
      f'  {hw_latency[best_acc_idx]*1e3:>8.1f}ms  {hw_memory[best_acc_idx]/1e6:>8.0f}MB')

# All methods ranked by tau
print(f'\n  -- All Methods Ranked by Kendall Tau --')
print(f'  {"Rank":>4s}  {"Method":>25s}  {"tau":>8s}  {"delta":>8s}')
print(f'  {"-"*50}')
all_ranked = [('hook_logdet', tau_b), ('layer_cka_weighted', tau_c)]
for m in SYSTEMS:
    sv = scores_sys[m][valid]
    vm = (sv != 0) & ~np.isnan(sv)
    t = stats.kendalltau(a[vm], sv[vm])[0] if vm.sum() > 1 else float('nan')
    all_ranked.append((m, t))
all_ranked.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
for rank, (method, tau) in enumerate(all_ranked, 1):
    diff = tau - tau_b if not np.isnan(tau) else float('nan')
    print(f'  {rank:>4d}  {method:>25s}  {tau:>+8.4f}  {diff:>+8.4f}')

print(f'\n{"="*70}')

# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

if not args.no_plots:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # ── CKA plot (3 panels) ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'CKA-Weighted Scoring -- {args.dataset} ({valid.sum()} architectures)',
                     fontsize=14, fontweight='bold')

        # 1. Scatter: Score vs Accuracy
        ax = axes[0]
        ax.scatter(a, sb, alpha=0.15, s=5, c='blue',
                   label=f'Baseline (tau={tau_b:.4f})')
        ax.scatter(a, sc, alpha=0.15, s=5, c='red',
                   label=f'CKA-weighted (tau={tau_c:.4f})')
        ax.set_xlabel('True Accuracy (%)')
        ax.set_ylabel('Score')
        ax.set_title('Score vs True Accuracy')
        ax.legend(fontsize=9)

        # 2. Top-K curve
        ax = axes[1]
        ks = list(range(1, min(201, valid.sum())))
        avg_b = [a[np.argsort(sb)[-k:]].mean() for k in ks]
        avg_c = [a[np.argsort(sc)[-k:]].mean() for k in ks]
        ax.plot(ks, avg_c, 'r-', lw=2, label='CKA-weighted')
        ax.plot(ks, avg_b, 'b-', lw=2, label='Baseline')
        ax.fill_between(ks, avg_b, avg_c, alpha=0.2, color='red')
        ax.set_xlabel('K')
        ax.set_ylabel('Avg Accuracy of Top-K (%)')
        ax.set_title('Top-K Selection Quality')
        ax.legend(fontsize=9)

        # 3. Bar chart: Kendall Tau
        ax = axes[2]
        methods = ['Baseline', 'CKA-weighted']
        taus = [tau_b, tau_c]
        colors = ['steelblue', 'indianred']
        bars = ax.bar(methods, taus, color=colors, edgecolor='black')
        ax.set_ylabel('Kendall Tau')
        ax.set_title('Ranking Correlation')
        for bar, tau in zip(bars, taus):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{tau:.4f}', ha='center', fontsize=11)
        ax.set_ylim(min(taus) - 0.05, max(taus) + 0.03)

        plt.tight_layout()
        os.makedirs('plots', exist_ok=True)
        cka_path = f'plots/cka_{args.dataset}.png'
        plt.savefig(cka_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\n  CKA plot saved: {cka_path}')

        # ── Systems table as image ──
        # Gather top-1 picks for each method
        sys_methods = {
            'CKA-weighted': sc,
            f'sys_params (a={alpha})': scores_sys[f'sys_params_{alpha}'][valid],
            f'sys_latency (a={alpha})': scores_sys[f'sys_latency_{alpha}'][valid],
            f'sys_memory (a={alpha})': scores_sys[f'sys_memory_{alpha}'][valid],
        }
        rows = []
        for name, s in sys_methods.items():
            idx = np.nanargmax(np.where(s != 0, s, -np.inf))
            rows.append([name,
                         f'{a[idx]:.2f}%',
                         f'{hw_params[valid][idx]/1e3:.0f}K',
                         f'{hw_latency[valid][idx]*1e3:.1f}ms',
                         f'{hw_memory[valid][idx]/1e6:.0f}MB'])

        fig_t, ax_t = plt.subplots(figsize=(10, 3))
        ax_t.axis('off')
        ax_t.set_title(f'Systems Analysis -- {args.dataset}: Top-1 Architecture per Method',
                       fontsize=13, fontweight='bold', pad=15)
        col_labels = ['Method', 'Accuracy', 'Params', 'Latency', 'Memory']
        table = ax_t.table(cellText=rows, colLabels=col_labels,
                           loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.6)
        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        # Alternate row colors
        for i in range(1, len(rows) + 1):
            color = '#D9E2F3' if i % 2 == 0 else 'white'
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(color)

        sys_path = f'plots/systems_{args.dataset}.png'
        plt.savefig(sys_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Systems table saved: {sys_path}')

    except Exception as e:
        print(f'  Plot generation failed: {e}')
