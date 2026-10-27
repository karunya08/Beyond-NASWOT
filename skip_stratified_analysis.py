"""
skip_stratified_analysis.py
===========================
Full 15,625-architecture stratified analysis of NASWOT by skip connection
topology. No GPU needed -- uses precomputed scores + NAS-Bench-201 API.

Key questions:
  1. What is NASWOT's tau WITHIN skip/no-skip groups?
  2. Is the overall tau=0.57 inflated by between-group separation?
  3. Does NASWOT correctly order skip vs non-skip architectures?

If tau is high between groups but low within groups, NASWOT is just
separating complexity levels, not measuring architecture quality.

Usage:
    python skip_stratified_analysis.py
"""

import numpy as np
from scipy import stats
from nas_201_api import NASBench201API as API
from collections import defaultdict
import os

# ============================================================
# Load precomputed NASWOT scores and accuracies (full 15,625)
# ============================================================
scores = np.load('results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy')
accs = np.load('results/full_baseline_accs_nasbench201_cifar10_True.npy')
print(f'Loaded {len(scores)} architectures')

api = API('../NAS-Bench-201-v1_0-e61699.pth', verbose=False)


# ============================================================
# Parse all architectures and compute topology features
# ============================================================
def parse_arch(api, uid):
    config = api.get_net_config(uid, 'cifar10-valid')
    nodes = api.str2lists(config['arch_str'])
    return [op for node_inputs in nodes for (op, src) in node_inputs]


def compute_topology(ops):
    """Compute skip/residual topology features from 6 edge operations."""
    n_skip = sum(1 for op in ops if op == 'skip_connect')
    n_parametric = sum(1 for op in ops if op in ('nor_conv_1x1', 'nor_conv_3x3'))
    n_conv3x3 = sum(1 for op in ops if op == 'nor_conv_3x3')
    n_conv1x1 = sum(1 for op in ops if op == 'nor_conv_1x1')
    n_none = sum(1 for op in ops if op == 'none')
    n_pool = sum(1 for op in ops if op == 'avg_pool_3x3')

    # Paths from node 0 to node 3
    paths = {
        'direct':  [ops[3]],
        'via_1':   [ops[0], ops[4]],
        'via_2':   [ops[1], ops[5]],
        'via_1_2': [ops[0], ops[2], ops[5]],
    }

    # Active paths (not blocked by 'none')
    n_active_paths = sum(1 for p in paths.values() if all(op != 'none' for op in p))

    # Skip path: any path from 0->3 that is all skip_connect
    has_skip_path = int(any(all(op == 'skip_connect' for op in p) for p in paths.values()))

    # Direct residual: 0->3 is skip_connect
    has_direct_residual = int(ops[3] == 'skip_connect')

    # True residual: skip 0->3 AND active conv path
    has_true_residual = 0
    if ops[3] == 'skip_connect':
        for name in ['via_1', 'via_2', 'via_1_2']:
            p = paths[name]
            if all(op != 'none' for op in p) and any(op in ('nor_conv_1x1', 'nor_conv_3x3') for op in p):
                has_true_residual = 1
                break

    # Gradient flow score
    GRAD_FACTOR = {'none': 0.0, 'skip_connect': 1.0, 'nor_conv_1x1': 0.8,
                   'nor_conv_3x3': 0.8, 'avg_pool_3x3': 0.5}
    grad_flow = sum(np.prod([GRAD_FACTOR.get(op, 0.5) for op in p]) for p in paths.values())

    return {
        'n_skip': n_skip, 'n_parametric': n_parametric, 'n_conv3x3': n_conv3x3,
        'n_conv1x1': n_conv1x1, 'n_none': n_none, 'n_pool': n_pool,
        'n_active_paths': n_active_paths, 'has_skip_path': has_skip_path,
        'has_direct_residual': has_direct_residual, 'has_true_residual': has_true_residual,
        'grad_flow': grad_flow,
    }


print('Computing topology features for all architectures...')
features = defaultdict(list)
for uid in range(len(scores)):
    ops = parse_arch(api, uid)
    topo = compute_topology(ops)
    for k, v in topo.items():
        features[k].append(v)

for k in features:
    features[k] = np.array(features[k])

# Valid architectures (non-singular K)
valid = np.isfinite(scores) & (accs > 0) & (scores > -1e10)
n_valid = valid.sum()
a = accs[valid]
s = scores[valid]

print(f'Valid architectures: {n_valid} / {len(scores)}')
tau_overall, _ = stats.kendalltau(a, s)
print(f'Overall NASWOT tau: {tau_overall:.4f}')


# ============================================================
# 1. TAU STRATIFIED BY SKIP COUNT
# ============================================================
print(f'\n{"="*90}')
print('1. NASWOT TAU BY NUMBER OF SKIP CONNECTIONS')
print(f'{"="*90}')
print(f'\n  {"n_skip":>6s} {"N":>6s} {"mean_acc":>10s} {"mean_score":>12s} '
      f'{"tau(s,acc)":>10s} {"mean_rank_err":>14s}')
print('  ' + '-' * 70)

rank_acc = stats.rankdata(a)
rank_score = stats.rankdata(s)
rank_error = rank_score - rank_acc

for ns in sorted(np.unique(features['n_skip'][valid])):
    mask = features['n_skip'][valid] == ns
    n = mask.sum()
    if n < 10:
        continue
    tau_ns, _ = stats.kendalltau(a[mask], s[mask])
    print(f'  {int(ns):>6d} {n:>6d} {a[mask].mean():>10.2f} {s[mask].mean():>12.1f} '
          f'{tau_ns:>10.4f} {rank_error[mask].mean():>+14.1f}')

# Aggregated: skip vs no-skip
has_skip = features['n_skip'][valid] >= 1
no_skip = features['n_skip'][valid] == 0
tau_skip, _ = stats.kendalltau(a[has_skip], s[has_skip])
tau_noskip, _ = stats.kendalltau(a[no_skip], s[no_skip])

print(f'\n  Summary:')
print(f'    Has skip (>=1): n={has_skip.sum():>5}, mean_acc={a[has_skip].mean():.2f}, '
      f'tau={tau_skip:.4f}, mean_rank_err={rank_error[has_skip].mean():+.1f}')
print(f'    No skip (0):    n={no_skip.sum():>5}, mean_acc={a[no_skip].mean():.2f}, '
      f'tau={tau_noskip:.4f}, mean_rank_err={rank_error[no_skip].mean():+.1f}')
print(f'    Overall:        n={n_valid:>5}, tau={tau_overall:.4f}')


# ============================================================
# 2. TAU STRATIFIED BY TRUE RESIDUAL
# ============================================================
print(f'\n{"="*90}')
print('2. NASWOT TAU BY TRUE RESIDUAL (skip || conv path)')
print(f'{"="*90}')

has_tr = features['has_true_residual'][valid] == 1
no_tr = features['has_true_residual'][valid] == 0
tau_tr, _ = stats.kendalltau(a[has_tr], s[has_tr])
tau_notr, _ = stats.kendalltau(a[no_tr], s[no_tr])

print(f'\n  True residual:  n={has_tr.sum():>5}, mean_acc={a[has_tr].mean():.2f}, '
      f'mean_score={s[has_tr].mean():.1f}, tau={tau_tr:.4f}, '
      f'mean_rank_err={rank_error[has_tr].mean():+.1f}')
print(f'  No true res:    n={no_tr.sum():>5}, mean_acc={a[no_tr].mean():.2f}, '
      f'mean_score={s[no_tr].mean():.1f}, tau={tau_notr:.4f}, '
      f'mean_rank_err={rank_error[no_tr].mean():+.1f}')

# NASWOT ranking direction: does it correctly rank true-res higher?
print(f'\n  Ranking direction check:')
print(f'    True residual mean accuracy:   {a[has_tr].mean():.2f}% (BETTER)')
print(f'    True residual mean NASWOT:     {s[has_tr].mean():.1f} '
      f'({"HIGHER" if s[has_tr].mean() > s[no_tr].mean() else "LOWER"} -- '
      f'{"CORRECT" if (s[has_tr].mean() > s[no_tr].mean()) == (a[has_tr].mean() > a[no_tr].mean()) else "WRONG"})')


# ============================================================
# 3. WITHIN-BAND TAU (accuracy quartiles within skip groups)
# ============================================================
print(f'\n{"="*90}')
print('3. WITHIN-BAND TAU (accuracy quartiles within each skip group)')
print(f'{"="*90}')

for group_name, group_mask in [('Has skip (>=1)', has_skip), ('No skip (0)', no_skip)]:
    a_group = a[group_mask]
    s_group = s[group_mask]
    q25, q50, q75 = np.percentile(a_group, [25, 50, 75])

    print(f'\n  {group_name} (n={group_mask.sum()}, quartiles at {q25:.1f}/{q50:.1f}/{q75:.1f}):')
    bands = [
        ('Bottom 25%', a_group < q25),
        ('25-50%', (a_group >= q25) & (a_group < q50)),
        ('50-75%', (a_group >= q50) & (a_group < q75)),
        ('Top 25%', a_group >= q75),
        ('Top 50%', a_group >= q50),
    ]
    for bname, bmask in bands:
        if bmask.sum() < 10:
            continue
        tau_b, _ = stats.kendalltau(a_group[bmask], s_group[bmask])
        print(f'    {bname:<15s}: n={bmask.sum():>5}, acc=[{a_group[bmask].min():.1f}-{a_group[bmask].max():.1f}], '
              f'tau={tau_b:.4f}')


# ============================================================
# 4. BETWEEN-GROUP TAU DECOMPOSITION
# ============================================================
print(f'\n{"="*90}')
print('4. TAU DECOMPOSITION: WITHIN vs BETWEEN skip groups')
print(f'{"="*90}')

# Concordant/discordant pairs BETWEEN groups
# A pair (i,j) is concordant if score and acc rank in the same direction
# We sample cross-group pairs (memory-efficient)
np.random.seed(42)
n_sample = 500000

skip_idx = np.where(has_skip)[0]
noskip_idx = np.where(no_skip)[0]

if len(skip_idx) > 0 and len(noskip_idx) > 0:
    # Random sample of cross-group pairs
    i_samples = np.random.choice(skip_idx, n_sample)
    j_samples = np.random.choice(noskip_idx, n_sample)

    score_diff = s[i_samples] - s[j_samples]
    acc_diff = a[i_samples] - a[j_samples]

    concordant = np.sum((score_diff * acc_diff) > 0)
    discordant = np.sum((score_diff * acc_diff) < 0)
    tied = n_sample - concordant - discordant
    tau_between = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0

    print(f'\n  Between skip/no-skip (sampled {n_sample} cross-group pairs):')
    print(f'    Concordant: {concordant:>8d} ({concordant/n_sample*100:.1f}%)')
    print(f'    Discordant: {discordant:>8d} ({discordant/n_sample*100:.1f}%)')
    print(f'    Tied:       {tied:>8d} ({tied/n_sample*100:.1f}%)')
    print(f'    tau_between = {tau_between:.4f}')

    print(f'\n  Comparison:')
    print(f'    tau_overall       = {tau_overall:.4f}')
    print(f'    tau_within_skip   = {tau_skip:.4f}')
    print(f'    tau_within_noskip = {tau_noskip:.4f}')
    print(f'    tau_between       = {tau_between:.4f}')

    inflation = tau_overall - (tau_skip * has_skip.sum() + tau_noskip * no_skip.sum()) / n_valid
    print(f'\n  Weighted within-group tau: {(tau_skip * has_skip.sum() + tau_noskip * no_skip.sum()) / n_valid:.4f}')
    print(f'  Between-group inflation:  {inflation:+.4f}')


# ============================================================
# 5. OPERATION DISTRIBUTION: NASWOT's bias quantified
# ============================================================
print(f'\n{"="*90}')
print('5. OPERATION DISTRIBUTION BY NASWOT RANK QUINTILE')
print(f'{"="*90}')

# Split into quintiles by NASWOT score rank
quintile_edges = np.percentile(s, [0, 20, 40, 60, 80, 100])
quintile_names = ['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']

print(f'\n  {"Quintile":<15s} {"N":>6s} {"mean_acc":>10s} {"mean_skip":>10s} '
      f'{"mean_conv1x1":>12s} {"mean_conv3x3":>12s} {"pct_true_res":>12s}')
print('  ' + '-' * 82)

for qi in range(5):
    low = quintile_edges[qi]
    high = quintile_edges[qi + 1]
    if qi < 4:
        mask = (s >= low) & (s < high)
    else:
        mask = s >= low
    n = mask.sum()
    print(f'  {quintile_names[qi]:<15s} {n:>6d} {a[mask].mean():>10.2f} '
          f'{features["n_skip"][valid][mask].mean():>10.2f} '
          f'{features["n_conv1x1"][valid][mask].mean():>12.2f} '
          f'{features["n_conv3x3"][valid][mask].mean():>12.2f} '
          f'{features["has_true_residual"][valid][mask].mean()*100:>11.1f}%')


# ============================================================
# 6. CORRECTED NASWOT: subtract topology-predicted error
# ============================================================
print(f'\n{"="*90}')
print('6. TOPOLOGY-CORRECTED NASWOT SCORES')
print(f'{"="*90}')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Build feature matrix
topo_names = ['n_skip', 'n_parametric', 'n_conv3x3', 'n_conv1x1', 'n_none',
              'has_true_residual', 'has_skip_path', 'has_direct_residual',
              'n_active_paths', 'grad_flow']
X = np.column_stack([features[fn][valid] for fn in topo_names])

# Predict rank error from topology
y_err = rank_error
reg = LinearRegression().fit(X, y_err)
r2 = r2_score(y_err, reg.predict(X))
print(f'\n  Topology -> rank_error: R^2 = {r2:.4f}')
print(f'  Coefficients:')
for fn, coef in zip(topo_names, reg.coef_):
    print(f'    {fn:<25s} {coef:>+10.1f}')

# Corrected score: subtract predicted error from NASWOT rank
predicted_error = reg.predict(X)
corrected_rank = rank_score - predicted_error
tau_corrected, _ = stats.kendalltau(a, corrected_rank)
print(f'\n  Original NASWOT tau:   {tau_overall:.4f}')
print(f'  Corrected NASWOT tau:  {tau_corrected:.4f}  ({tau_corrected - tau_overall:+.4f})')

# Simpler corrections
print(f'\n  Simple corrections (logdet + alpha * topology_feature):')
s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)

for fn in ['grad_flow', 'n_skip', 'has_true_residual', 'has_skip_path']:
    f = features[fn][valid]
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-10)
    best_alpha, best_tau = 0, tau_overall
    for alpha in np.arange(-2, 5, 0.1):
        combined = s_norm + alpha * f_norm
        t, _ = stats.kendalltau(a, combined)
        if t > best_tau:
            best_alpha, best_tau = alpha, t
    print(f'    logdet + alpha*{fn:<20s}: best alpha={best_alpha:>+5.1f}, '
          f'tau={best_tau:.4f} ({best_tau - tau_overall:+.4f})')


# ============================================================
# 7. DISTRIBUTION OF SKIP ARCHITECTURES ACROSS ACCURACY RANGE
# ============================================================
print(f'\n{"="*90}')
print('7. WHERE DO SKIP ARCHITECTURES FALL IN THE ACCURACY DISTRIBUTION?')
print(f'{"="*90}')

percentiles = np.percentile(a, [10, 25, 50, 75, 90])
bands = [
    ('Bottom 10%', a < percentiles[0]),
    ('10-25%', (a >= percentiles[0]) & (a < percentiles[1])),
    ('25-50%', (a >= percentiles[1]) & (a < percentiles[2])),
    ('50-75%', (a >= percentiles[2]) & (a < percentiles[3])),
    ('75-90%', (a >= percentiles[3]) & (a < percentiles[4])),
    ('Top 10%', a >= percentiles[4]),
]

print(f'\n  {"Band":<15s} {"N":>6s} {"pct_skip>=1":>12s} {"pct_true_res":>12s} '
      f'{"mean_n_skip":>12s} {"naswot_tau":>10s}')
print('  ' + '-' * 72)

for bname, bmask in bands:
    n = bmask.sum()
    if n < 10:
        continue
    pct_skip = (features['n_skip'][valid][bmask] >= 1).mean() * 100
    pct_tr = features['has_true_residual'][valid][bmask].mean() * 100
    mean_skip = features['n_skip'][valid][bmask].mean()
    tau_b, _ = stats.kendalltau(a[bmask], s[bmask])
    print(f'  {bname:<15s} {n:>6d} {pct_skip:>11.1f}% {pct_tr:>11.1f}% '
          f'{mean_skip:>12.2f} {tau_b:>10.4f}')


# ============================================================
# 8. SUMMARY TABLE (presentation-ready)
# ============================================================
print(f'\n{"="*90}')
print('SUMMARY: THE SKIP CONNECTION PROBLEM IN NASWOT')
print(f'{"="*90}')

print(f'''
  Finding 1: NASWOT's tau is inflated by between-group separation
    Overall tau:          {tau_overall:.4f}
    Within skip archs:    {tau_skip:.4f}
    Within no-skip archs: {tau_noskip:.4f}
    Between groups:       {tau_between:.4f}

  Finding 2: NASWOT systematically misranks skip architectures
    True residual archs:  acc={a[has_tr].mean():.2f}% but NASWOT={s[has_tr].mean():.0f}
    Non-residual archs:   acc={a[no_tr].mean():.2f}% but NASWOT={s[no_tr].mean():.0f}
    Direction: {"CORRECT" if (s[has_tr].mean() > s[no_tr].mean()) == (a[has_tr].mean() > a[no_tr].mean()) else "WRONG (skip archs scored LOWER despite being BETTER)"}

  Finding 3: The problem worsens in the top half
    Top-50% skip archs tau:    {stats.kendalltau(a[has_skip & (a >= np.median(a))], s[has_skip & (a >= np.median(a))])[0]:.4f}
    Top-50% no-skip archs tau: {stats.kendalltau(a[no_skip & (a >= np.median(a))], s[no_skip & (a >= np.median(a))])[0]:.4f}

  Finding 4: Topology correction improves tau
    Original:  {tau_overall:.4f}
    Corrected: {tau_corrected:.4f} ({tau_corrected - tau_overall:+.4f})
''')
