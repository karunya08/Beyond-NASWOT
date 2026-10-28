"""
search_pipeline.py
==================
Two-stage NAS diagnostic pipeline: Primary metric + Gradient Consistency.

Uses pre-computed primary scores (NASWOT or Feature Diversity, all 15,625
architectures) and only computes GC for the top-K pool from stage 1.

Stage 1: Load pre-computed primary scores -> take top-K as quality pool.
Stage 2: Compute GC for the top-K pool -> re-rank by GC to select
         architectures with skip connections (faster convergence).

Usage:
    # Default: NASWOT top quartile, re-rank by GC
    python search_pipeline.py --trainval --GPU 0

    # Feature diversity as primary
    python search_pipeline.py --trainval --GPU 0 --primary feat_div

    # Custom pool size
    python search_pipeline.py --trainval --GPU 0 --top_k 500
"""

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import torch
import torch.nn as nn
from models import get_cell_based_tiny_net
from tqdm import trange
import time

parser = argparse.ArgumentParser(description='Two-stage NAS: Primary + GC')
parser.add_argument('--data_loc', default='../cifardata/', type=str)
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--save_loc', default='results', type=str)
parser.add_argument('--nasspace', default='nasbench201', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--primary', default='naswot', choices=['naswot', 'feat_div'],
                    help='primary metric for stage 1 filtering')
parser.add_argument('--top_k', default=0, type=int,
                    help='exact pool size from stage 1 (overrides --quartile)')
parser.add_argument('--quartile', default=1, type=int, choices=[1, 2, 3, 4],
                    help='which quartile to use as pool (1=top 25%%, 2=top 50%%, etc.)')
parser.add_argument('--stem_out_channels', default=16, type=int)
parser.add_argument('--num_stacks', default=3, type=int)
parser.add_argument('--num_modules_per_stack', default=3, type=int)
parser.add_argument('--num_labels', default=1, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'ImageNet16-120': 120}
n_classes = NUM_CLASSES.get(args.dataset, 10)

searchspace = nasspace.get_search_space(args)
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval,
                                  args.batch_size, 'none', 1, args)


# ============================================================
# Load pre-computed primary scores
# ============================================================
SCORE_FILES = {
    'naswot': 'results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy',
    'feat_div': 'results/full_feat_div_scores.npy',
}
accs_file = 'results/full_baseline_accs_nasbench201_cifar10_True.npy'
scores_file = SCORE_FILES[args.primary]
primary_name = 'NASWOT' if args.primary == 'naswot' else 'FeatDiv'

print(f'Loading pre-computed {primary_name} scores from {scores_file}')
primary_scores = np.load(scores_file)
accs = np.load(accs_file)
print(f'  Loaded {len(primary_scores)} architectures')

# Valid architectures (finite score, positive accuracy)
valid = np.isfinite(primary_scores) & (primary_scores > 0) & (accs > 0)
valid_idx = np.where(valid)[0]
n_valid = valid.sum()
print(f'  Valid: {n_valid}')


# ============================================================
# Parse topology (free, no GPU)
# ============================================================
print('Parsing architecture topology...')
all_skips = np.zeros(len(primary_scores), dtype=int)
all_tres = np.zeros(len(primary_scores), dtype=int)

for uid in valid_idx:
    config = searchspace.api.get_net_config(uid, 'cifar10-valid')
    nodes = searchspace.api.str2lists(config['arch_str'])
    ops = [op for node_inputs in nodes for (op, src) in node_inputs]
    all_skips[uid] = sum(1 for op in ops if op == 'skip_connect')
    # True residual: skip on node2->node3 edge AND conv path exists
    if ops[3] == 'skip_connect':
        paths = {'a': [ops[0], ops[4]], 'b': [ops[1], ops[5]],
                 'c': [ops[0], ops[2], ops[5]]}
        for pops in paths.values():
            if (all(op != 'none' for op in pops) and
                any(op in ('nor_conv_1x1', 'nor_conv_3x3') for op in pops)):
                all_tres[uid] = 1
                break


# ============================================================
# STAGE 1: Primary metric top-K
# ============================================================
sorted_by_primary = valid_idx[np.argsort(-primary_scores[valid_idx])]
if args.top_k > 0:
    top_k = min(args.top_k, n_valid)
else:
    top_k = (n_valid * args.quartile) // 4
    print(f'  Using quartile {args.quartile} -> top {top_k} architectures ({args.quartile * 25}%)')
pool_uids = sorted_by_primary[:top_k]

print(f'\n{"="*70}')
print(f'STAGE 1: {primary_name} top-{top_k}')
print(f'{"="*70}')
print(f'  Mean accuracy:  {accs[pool_uids].mean():.2f}%')
print(f'  Mean skips:     {all_skips[pool_uids].mean():.2f}')
print(f'  % with skip:    {100 * (all_skips[pool_uids] >= 1).mean():.1f}%')
print(f'  % true residual: {100 * all_tres[pool_uids].mean():.1f}%')

# Skip distribution
for ns in sorted(set(all_skips[pool_uids])):
    mask = all_skips[pool_uids] == ns
    n = mask.sum()
    ma = accs[pool_uids[mask]].mean()
    print(f'    {ns} skips: n={n:>5d}, mean_acc={ma:.2f}%')


# ============================================================
# STAGE 2: Compute GC for top-K pool only
# ============================================================
print(f'\n{"="*70}')
print(f'STAGE 2: Computing Gradient Consistency for {top_k} architectures...')
print(f'{"="*70}')


def compute_grad_consistency(network, xA, yA, xB, yB):
    """Gradient consistency: cosine similarity of gradients on two batches."""
    network.train()
    criterion = nn.CrossEntropyLoss()

    network.zero_grad()
    logits_A, _ = network(xA)
    criterion(logits_A, yA).backward()
    g_A = torch.cat([p.grad.flatten() for p in network.parameters()
                     if p.grad is not None])

    network.zero_grad()
    logits_B, _ = network(xB)
    criterion(logits_B, yB).backward()
    g_B = torch.cat([p.grad.flatten() for p in network.parameters()
                     if p.grad is not None])

    network.zero_grad()
    return (g_A @ g_B).item() / (g_A.norm().item() * g_B.norm().item() + 1e-10)


# Load two data batches
data_iter = iter(train_loader)
xA, yA = next(data_iter)
xB, yB = next(data_iter)
xA, yA = xA.to(device), yA.to(device)
xB, yB = xB.to(device), yB.to(device)

gc_scores = np.zeros(top_k)
gc_start = time.time()

for i, uid in enumerate(pool_uids):
    try:
        config = searchspace.api.get_net_config(uid, 'cifar10-valid')
        config['num_classes'] = n_classes
        network = get_cell_based_tiny_net(config)
        for m in network.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False
        network = network.to(device)

        gc_scores[i] = compute_grad_consistency(network, xA, yA, xB, yB)

        del network
        torch.cuda.empty_cache()
    except Exception as e:
        gc_scores[i] = 0.

    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - gc_start
        rate = (i + 1) / elapsed
        remaining = (top_k - i - 1) / rate
        print(f'  [{i+1}/{top_k}] gc={gc_scores[i]:.4f}, skips={all_skips[uid]} '
              f'({rate:.1f} arch/s, ETA {remaining/60:.1f} min)', flush=True)


# ============================================================
# RESULTS: Compare NASWOT pick vs GC-reranked pick
# ============================================================
print(f'\n{"="*70}')
print('RESULTS')
print(f'{"="*70}')

# NASWOT's #1 pick (best NASWOT score)
naswot_best = pool_uids[0]
print(f'\n{primary_name} #1 pick:')
print(f'  uid={naswot_best}, acc={accs[naswot_best]:.2f}%, '
      f'skips={all_skips[naswot_best]}, {primary_name}={primary_scores[naswot_best]:.2f}')

# GC's #1 pick (highest GC in pool)
gc_best_idx = np.argmax(gc_scores)
gc_best = pool_uids[gc_best_idx]
print(f'\nGC re-ranked #1 pick (from top-{top_k} pool):')
print(f'  uid={gc_best}, acc={accs[gc_best]:.2f}%, '
      f'skips={all_skips[gc_best]}, gc={gc_scores[gc_best_idx]:.4f}')

# Compare top-10 from each method
n_compare = min(10, top_k)

print(f'\nTop-{n_compare} by {primary_name}:')
primary_top = pool_uids[:n_compare]
print(f'  Mean acc:   {accs[primary_top].mean():.2f}%')
print(f'  Mean skips: {all_skips[primary_top].mean():.2f}')
print(f'  % with skip: {100 * (all_skips[primary_top] >= 1).mean():.1f}%')

print(f'\nTop-{n_compare} by GC (from pool):')
gc_top_idx = np.argsort(-gc_scores)[:n_compare]
gc_top = pool_uids[gc_top_idx]
print(f'  Mean acc:   {accs[gc_top].mean():.2f}%')
print(f'  Mean skips: {all_skips[gc_top].mean():.2f}')
print(f'  % with skip: {100 * (all_skips[gc_top] >= 1).mean():.1f}%')

# Correlation between GC and skip count in the pool
from scipy import stats

tau_gc_skip, p_gc_skip = stats.kendalltau(gc_scores, all_skips[pool_uids])
tau_gc_acc, p_gc_acc = stats.kendalltau(gc_scores, accs[pool_uids])

print(f'\nCorrelations within top-{top_k} pool:')
print(f'  GC vs skip count: tau={tau_gc_skip:.4f}, p={p_gc_skip:.4f}')
print(f'  GC vs accuracy:   tau={tau_gc_acc:.4f}, p={p_gc_acc:.4f}')

# Split pool by GC median -- do high-GC architectures have more skips?
gc_median = np.median(gc_scores)
high_gc = gc_scores >= gc_median
low_gc = gc_scores < gc_median

print(f'\nPool split by GC median ({gc_median:.4f}):')
print(f'  High GC (n={high_gc.sum()}): mean_acc={accs[pool_uids[high_gc]].mean():.2f}%, '
      f'mean_skips={all_skips[pool_uids[high_gc]].mean():.2f}, '
      f'% true_res={100*all_tres[pool_uids[high_gc]].mean():.1f}%')
print(f'  Low GC  (n={low_gc.sum()}):  mean_acc={accs[pool_uids[low_gc]].mean():.2f}%, '
      f'mean_skips={all_skips[pool_uids[low_gc]].mean():.2f}, '
      f'% true_res={100*all_tres[pool_uids[low_gc]].mean():.1f}%')

# t-tests
t_acc, p_acc = stats.ttest_ind(accs[pool_uids[high_gc]], accs[pool_uids[low_gc]])
t_skip, p_skip = stats.ttest_ind(all_skips[pool_uids[high_gc]], all_skips[pool_uids[low_gc]])
print(f'  Accuracy diff:   t={t_acc:.3f}, p={p_acc:.4f}')
print(f'  Skip count diff: t={t_skip:.3f}, p={p_skip:.4f}')


# ============================================================
# Compare with TRUE top-K
# ============================================================
sorted_by_acc = valid_idx[np.argsort(-accs[valid_idx])]
true_top = sorted_by_acc[:top_k]

print(f'\n{"="*70}')
print(f'COMPARISON WITH GROUND TRUTH (true top-{top_k} by accuracy)')
print(f'{"="*70}')
print(f'  TRUE top-{top_k}:   mean_acc={accs[true_top].mean():.2f}%, '
      f'mean_skips={all_skips[true_top].mean():.2f}, '
      f'% true_res={100*all_tres[true_top].mean():.1f}%')
print(f'  {primary_name} top-{top_k}: mean_acc={accs[pool_uids].mean():.2f}%, '
      f'mean_skips={all_skips[pool_uids].mean():.2f}, '
      f'% true_res={100*all_tres[pool_uids].mean():.1f}%')
print(f'  GC top half:     mean_acc={accs[pool_uids[high_gc]].mean():.2f}%, '
      f'mean_skips={all_skips[pool_uids[high_gc]].mean():.2f}, '
      f'% true_res={100*all_tres[pool_uids[high_gc]].mean():.1f}%')


# Save results
os.makedirs(args.save_loc, exist_ok=True)
save_data = {
    'pool_uids': pool_uids,
    'gc_scores': gc_scores,
    'primary_scores': primary_scores[pool_uids],
    'accs': accs[pool_uids],
    'skips': all_skips[pool_uids],
    'true_res': all_tres[pool_uids],
    'top_k': top_k,
}
fname = f'{args.save_loc}/pipeline_{args.primary}_topk{top_k}_gc.npy'
np.savez(fname, **save_data)
print(f'\nResults saved to {fname}')

