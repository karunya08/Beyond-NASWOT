"""
failure_attribution.py
======================
Analyze WHY NASWOT misjudges specific architectures.

Loads the overrated/underrated failure sets from failure_analysis.py
and queries NAS-Bench-201 for architecture structure, then compares
operation distributions, param counts, and structural patterns.

Run from project root:
    python experiments/failure_attribution.py --api_loc ../NAS-Bench-201-v1_0-e61699.pth

Prerequisite: Run failure_analysis.py first to generate failure_sets/
    python experiments/failure_analysis.py \
        --scores results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy \
        --accs results/full_baseline_accs_nasbench201_cifar10_True.npy
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from collections import Counter
from nas_201_api import NASBench201API as API

parser = argparse.ArgumentParser(description='NASWOT Failure Attribution')
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--failure_dir', default='failure_sets', type=str)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load failure sets
# ---------------------------------------------------------------------------
overrated_uids = np.load(os.path.join(args.failure_dir, 'overrated_uids.npy'))
underrated_uids = np.load(os.path.join(args.failure_dir, 'underrated_uids.npy'))
all_uids = np.load(os.path.join(args.failure_dir, 'all_valid_uids.npy'))
all_scores = np.load(os.path.join(args.failure_dir, 'all_valid_scores.npy'))
all_accs = np.load(os.path.join(args.failure_dir, 'all_valid_accs.npy'))

print(f'Overrated:  {len(overrated_uids)} architectures')
print(f'Underrated: {len(underrated_uids)} architectures')
print(f'Total valid: {len(all_uids)} architectures')

# ---------------------------------------------------------------------------
# Load API
# ---------------------------------------------------------------------------
print(f'\nLoading NAS-Bench-201 API...')
api = API(args.api_loc, verbose=False)

# NAS-Bench-201 operations
OPS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
N_EDGES = 6  # 3-node cell with 6 edges


def parse_arch_str(arch_str):
    """Parse NAS-Bench-201 arch string into list of op names."""
    ops = []
    for part in arch_str.split('+'):
        part = part.strip('|')
        for token in part.split('|'):
            token = token.strip()
            if '~' in token:
                op_name = token.split('~')[0]
                ops.append(op_name)
    return ops


def get_arch_ops(uid):
    """Get the list of operations for an architecture."""
    config = api.get_net_config(int(uid), 'cifar10-valid')
    return parse_arch_str(config['arch_str']), config['arch_str']


def op_distribution(uids):
    """Count operation frequency across a set of architectures."""
    total_ops = Counter()
    for uid in uids:
        ops, _ = get_arch_ops(uid)
        for op in ops:
            total_ops[op] += 1
    n = len(uids) * N_EDGES  # total op slots
    return {op: total_ops.get(op, 0) / n for op in OPS}, total_ops


# ---------------------------------------------------------------------------
# Operation distribution comparison
# ---------------------------------------------------------------------------
print('\n--- Operation Distribution ---')
print(f'{"Op":<16} {"Overrated":>10} {"Underrated":>10} {"All":>10} {"Over-All":>10}')
print('-' * 60)

over_dist, over_counts = op_distribution(overrated_uids)
under_dist, under_counts = op_distribution(underrated_uids)
all_dist, all_counts = op_distribution(all_uids)

for op in OPS:
    diff = over_dist[op] - all_dist[op]
    print(f'{op:<16} {over_dist[op]:>9.1%} {under_dist[op]:>9.1%} '
          f'{all_dist[op]:>9.1%} {diff:>+9.1%}')

# ---------------------------------------------------------------------------
# Specific patterns: none and skip_connect counts per architecture
# ---------------------------------------------------------------------------
print('\n--- None + Skip Counts per Architecture ---')

def count_ops_per_arch(uids):
    """For each arch, count none, skip, conv1x1, conv3x3, pool ops."""
    records = []
    for uid in uids:
        ops, arch_str = get_arch_ops(uid)
        records.append({
            'uid': int(uid),
            'none': ops.count('none'),
            'skip': ops.count('skip_connect'),
            'conv1x1': ops.count('nor_conv_1x1'),
            'conv3x3': ops.count('nor_conv_3x3'),
            'pool': ops.count('avg_pool_3x3'),
            'arch_str': arch_str,
        })
    return records


print('Computing op counts for overrated set...')
over_records = count_ops_per_arch(overrated_uids)
print('Computing op counts for underrated set...')
under_records = count_ops_per_arch(underrated_uids)

over_none = np.array([r['none'] for r in over_records])
over_skip = np.array([r['skip'] for r in over_records])
over_conv3 = np.array([r['conv3x3'] for r in over_records])
under_none = np.array([r['none'] for r in under_records])
under_skip = np.array([r['skip'] for r in under_records])
under_conv3 = np.array([r['conv3x3'] for r in under_records])

print(f'{"Metric":<25} {"Overrated":>10} {"Underrated":>10}')
print('-' * 48)
print(f'{"avg none ops":<25} {over_none.mean():>10.2f} {under_none.mean():>10.2f}')
print(f'{"avg skip ops":<25} {over_skip.mean():>10.2f} {under_skip.mean():>10.2f}')
print(f'{"avg conv3x3 ops":<25} {over_conv3.mean():>10.2f} {under_conv3.mean():>10.2f}')
print(f'{"avg none+skip":<25} {(over_none+over_skip).mean():>10.2f} '
      f'{(under_none+under_skip).mean():>10.2f}')
print(f'{"% with >=2 none ops":<25} {(over_none>=2).mean():>9.1%} '
      f'{(under_none>=2).mean():>9.1%}')
print(f'{"% with 0 none ops":<25} {(over_none==0).mean():>9.1%} '
      f'{(under_none==0).mean():>9.1%}')

# ---------------------------------------------------------------------------
# Show example architectures
# ---------------------------------------------------------------------------
print('\n--- Example Overrated Architectures (high score, low accuracy) ---')
# Find the overrated with highest score-to-acc gap
over_with_accs = []
for r in over_records:
    uid = r['uid']
    idx = np.where(all_uids == uid)[0]
    if len(idx) > 0:
        r['score'] = all_scores[idx[0]]
        r['acc'] = all_accs[idx[0]]
        over_with_accs.append(r)

over_with_accs.sort(key=lambda r: r['acc'])
for r in over_with_accs[:10]:
    print(f"  uid={r['uid']:5d}  score={r['score']:.0f}  acc={r['acc']:.1f}%  "
          f"none={r['none']} skip={r['skip']} conv3={r['conv3x3']}  {r['arch_str']}")

print('\n--- Example Underrated Architectures (low score, high accuracy) ---')
under_with_accs = []
for r in under_records:
    uid = r['uid']
    idx = np.where(all_uids == uid)[0]
    if len(idx) > 0:
        r['score'] = all_scores[idx[0]]
        r['acc'] = all_accs[idx[0]]
        under_with_accs.append(r)

under_with_accs.sort(key=lambda r: -r['acc'])
for r in under_with_accs[:10]:
    print(f"  uid={r['uid']:5d}  score={r['score']:.0f}  acc={r['acc']:.1f}%  "
          f"none={r['none']} skip={r['skip']} conv3={r['conv3x3']}  {r['arch_str']}")

# ---------------------------------------------------------------------------
# Partial correlation: does score predict accuracy after controlling for #ops?
# ---------------------------------------------------------------------------
print('\n--- Score vs Accuracy, controlling for architecture complexity ---')
from scipy import stats

# Use conv3x3 count as a proxy for complexity (most parameters)
print('Computing op counts for all architectures (this takes a minute)...')
all_conv3 = np.zeros(len(all_uids))
all_none_count = np.zeros(len(all_uids))
for i, uid in enumerate(all_uids):
    ops, _ = get_arch_ops(uid)
    all_conv3[i] = ops.count('nor_conv_3x3')
    all_none_count[i] = ops.count('none')
    if (i + 1) % 2000 == 0:
        print(f'  {i+1}/{len(all_uids)}...')

# Correlation: score vs accuracy
tau_score_acc, _ = stats.kendalltau(all_scores, all_accs)
# Correlation: #conv3x3 vs accuracy
tau_conv3_acc, _ = stats.kendalltau(all_conv3, all_accs)
# Correlation: #none vs accuracy
tau_none_acc, _ = stats.kendalltau(all_none_count, all_accs)
# Correlation: score vs #conv3x3
tau_score_conv3, _ = stats.kendalltau(all_scores, all_conv3)

print(f'  tau(score, acc)     = {tau_score_acc:.4f}')
print(f'  tau(#conv3x3, acc)  = {tau_conv3_acc:.4f}')
print(f'  tau(#none, acc)     = {tau_none_acc:.4f}')
print(f'  tau(score, #conv3x3)= {tau_score_conv3:.4f}')

# Simple question: is NASWOT just counting conv3x3 ops?
# If tau(score, conv3x3) is high, NASWOT is mostly a complexity proxy.
if abs(tau_score_conv3) > 0.4:
    print('  >> NASWOT score correlates strongly with #conv3x3 -- it may be a complexity proxy')
else:
    print('  >> NASWOT score is NOT just a complexity proxy')

print('\nDone.')
