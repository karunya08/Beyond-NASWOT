"""
failure_analysis.py
===================
Identify and characterize NASWOT failure cases on NAS-Bench-201.

False positives: architectures ranked highly by NASWOT but with low actual accuracy.
False negatives: architectures ranked poorly by NASWOT but with high actual accuracy.

Usage
-----
python failure_analysis.py --scores results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy \
                           --accs results/full_baseline_accs_nasbench201_cifar10_True.npy
"""

import argparse
import numpy as np
import os
from scipy import stats

parser = argparse.ArgumentParser(description='NASWOT Failure Analysis')
parser.add_argument('--scores', required=True, type=str,
                    help='path to scores .npy file')
parser.add_argument('--accs', required=True, type=str,
                    help='path to accuracies .npy file')
parser.add_argument('--save_dir', default='failure_sets', type=str,
                    help='directory to save failure UIDs')
parser.add_argument('--fp_score_pct', default=10, type=float,
                    help='false positives: top X%% by score (default: 10)')
parser.add_argument('--fp_acc_pct', default=30, type=float,
                    help='false positives: bottom Y%% by accuracy (default: 30)')
parser.add_argument('--fn_score_pct', default=30, type=float,
                    help='false negatives: bottom X%% by score (default: 30)')
parser.add_argument('--fn_acc_pct', default=10, type=float,
                    help='false negatives: top Y%% by accuracy (default: 10)')
args = parser.parse_args()

scores_file = args.scores
accs_file = args.accs

print(f'Scores: {scores_file}')
print(f'Accs:   {accs_file}')

scores = np.load(scores_file)
accs = np.load(accs_file)

# ---------------------------------------------------------------------------
# Filter to valid (scored, finite, with accuracy) architectures
# ---------------------------------------------------------------------------
valid = (scores != 0) & np.isfinite(scores) & (accs > 0)
n_valid = valid.sum()
n_total = len(scores)
n_inf = np.isinf(scores).sum()

print(f'\nTotal architectures: {n_total}')
print(f'Valid (scored, finite, acc > 0): {n_valid}')
print(f'Inf scores (singular K): {n_inf}')
print(f'Unscored (zero): {n_total - n_valid - n_inf}')

if n_valid < 100:
    print(f'\nERROR: Only {n_valid} valid architectures. Need a complete run.')
    print('Run: python score_networks.py --trainval --augtype none --repeat 1 '
          '--score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 '
          '--GPU 0 --dataset cifar10 --method masks --mask_type relu '
          '--save_string full_baseline')
    exit(1)

# Work only with valid entries
valid_uids = np.where(valid)[0]
valid_scores = scores[valid]
valid_accs = accs[valid]

# Overall Kendall tau
tau, p = stats.kendalltau(valid_accs, valid_scores)
print(f'\nOverall Kendall-tau: {tau:.4f} (p={p:.2e}, n={n_valid})')

# ---------------------------------------------------------------------------
# Percentile thresholds
# ---------------------------------------------------------------------------
score_top_thresh = np.percentile(valid_scores, 100 - args.fp_score_pct)
score_bot_thresh = np.percentile(valid_scores, args.fn_score_pct)
acc_top_thresh = np.percentile(valid_accs, 100 - args.fn_acc_pct)
acc_bot_thresh = np.percentile(valid_accs, args.fp_acc_pct)

print(f'\n--- Thresholds ---')
print(f'Score top {args.fp_score_pct}%: >= {score_top_thresh:.2f}')
print(f'Score bottom {args.fn_score_pct}%: <= {score_bot_thresh:.2f}')
print(f'Acc top {args.fn_acc_pct}%: >= {acc_top_thresh:.2f}%')
print(f'Acc bottom {args.fp_acc_pct}%: <= {acc_bot_thresh:.2f}%')

# ---------------------------------------------------------------------------
# Identify failures
# ---------------------------------------------------------------------------

# False positives: NASWOT says great (top score), reality says bad (low acc)
fp_mask = (valid_scores >= score_top_thresh) & (valid_accs <= acc_bot_thresh)
fp_uids = valid_uids[fp_mask]
fp_scores = valid_scores[fp_mask]
fp_accs = valid_accs[fp_mask]

# False negatives: NASWOT says bad (low score), reality says great (high acc)
fn_mask = (valid_scores <= score_bot_thresh) & (valid_accs >= acc_top_thresh)
fn_uids = valid_uids[fn_mask]
fn_scores = valid_scores[fn_mask]
fn_accs = valid_accs[fn_mask]

# True positives: NASWOT says great, reality agrees
tp_mask = (valid_scores >= score_top_thresh) & (valid_accs >= acc_top_thresh)
tp_uids = valid_uids[tp_mask]

# True negatives: NASWOT says bad, reality agrees
tn_mask = (valid_scores <= score_bot_thresh) & (valid_accs <= acc_bot_thresh)
tn_uids = valid_uids[tn_mask]

print(f'\n--- Failure Sets ---')
print(f'False positives (high score, low acc): {len(fp_uids)}')
print(f'False negatives (low score, high acc): {len(fn_uids)}')
print(f'True positives  (high score, high acc): {len(tp_uids)}')
print(f'True negatives  (low score, low acc):  {len(tn_uids)}')

# ---------------------------------------------------------------------------
# Characterize failures
# ---------------------------------------------------------------------------
def describe_set(name, uids, set_scores, set_accs):
    if len(uids) == 0:
        print(f'\n  {name}: empty')
        return
    print(f'\n  {name} ({len(uids)} architectures):')
    print(f'    Score: {set_scores.min():.1f} / {np.median(set_scores):.1f} / {set_scores.max():.1f}'
          f' (min/med/max)')
    print(f'    Acc:   {set_accs.min():.2f} / {np.median(set_accs):.2f} / {set_accs.max():.2f}%'
          f' (min/med/max)')
    # Show worst offenders
    if name == 'FALSE POSITIVES':
        # Sort by acc ascending (worst accuracy first)
        order = np.argsort(set_accs)
    else:
        # Sort by acc descending (best accuracy first)
        order = np.argsort(-set_accs)
    n_show = min(10, len(uids))
    print(f'    Top {n_show} worst offenders:')
    for j in range(n_show):
        idx = order[j]
        print(f'      uid={uids[idx]:5d}  score={set_scores[idx]:.1f}  acc={set_accs[idx]:.2f}%')


print('\n--- Failure Characterization ---')
describe_set('FALSE POSITIVES', fp_uids, fp_scores, fp_accs)
describe_set('FALSE NEGATIVES', fn_uids, fn_scores, fn_accs)
describe_set('TRUE POSITIVES', tp_uids, valid_scores[tp_mask], valid_accs[tp_mask])
describe_set('TRUE NEGATIVES', tn_uids, valid_scores[tn_mask], valid_accs[tn_mask])

# ---------------------------------------------------------------------------
# Score distribution comparison
# ---------------------------------------------------------------------------
print('\n--- Score Distribution by Accuracy Quartile ---')
acc_quartiles = np.percentile(valid_accs, [25, 50, 75])
q_labels = [
    f'Q1 (acc <= {acc_quartiles[0]:.1f}%)',
    f'Q2 ({acc_quartiles[0]:.1f} < acc <= {acc_quartiles[1]:.1f}%)',
    f'Q3 ({acc_quartiles[1]:.1f} < acc <= {acc_quartiles[2]:.1f}%)',
    f'Q4 (acc > {acc_quartiles[2]:.1f}%)',
]
for qi, ql in enumerate(q_labels):
    if qi == 0:
        mask = valid_accs <= acc_quartiles[0]
    elif qi == 3:
        mask = valid_accs > acc_quartiles[2]
    else:
        mask = (valid_accs > acc_quartiles[qi-1]) & (valid_accs <= acc_quartiles[qi])
    qs = valid_scores[mask]
    print(f'  {ql}: n={mask.sum():5d}  score={qs.mean():.1f} +/- {qs.std():.1f}')

# ---------------------------------------------------------------------------
# Rank inversions: where does NASWOT actually get the ordering wrong?
# ---------------------------------------------------------------------------
print('\n--- Rank Inversion Analysis ---')

# Rank all architectures by score and by accuracy
score_ranks = stats.rankdata(valid_scores)
acc_ranks = stats.rankdata(valid_accs)
rank_diff = score_ranks - acc_ranks  # positive = NASWOT overrates, negative = underrates

# Overrated: NASWOT ranks much higher than accuracy warrants
overrated_thresh = np.percentile(rank_diff, 95)  # top 5% most overrated
underrated_thresh = np.percentile(rank_diff, 5)   # top 5% most underrated

overrated_mask = rank_diff >= overrated_thresh
underrated_mask = rank_diff <= underrated_thresh

overrated_uids = valid_uids[overrated_mask]
underrated_uids = valid_uids[underrated_mask]

print(f'Overrated (score rank >> acc rank):  {overrated_mask.sum()} architectures')
print(f'  Rank diff threshold: >= {overrated_thresh:.0f}')
print(f'  Score: {valid_scores[overrated_mask].mean():.1f} +/- {valid_scores[overrated_mask].std():.1f}')
print(f'  Acc:   {valid_accs[overrated_mask].mean():.2f} +/- {valid_accs[overrated_mask].std():.2f}%')

print(f'Underrated (score rank << acc rank): {underrated_mask.sum()} architectures')
print(f'  Rank diff threshold: <= {underrated_thresh:.0f}')
print(f'  Score: {valid_scores[underrated_mask].mean():.1f} +/- {valid_scores[underrated_mask].std():.1f}')
print(f'  Acc:   {valid_accs[underrated_mask].mean():.2f} +/- {valid_accs[underrated_mask].std():.2f}%')

# Where do rank inversions concentrate? Break by accuracy band.
print('\n--- Rank Inversions by Accuracy Band ---')
print('  (mean |rank_diff|: how wrong NASWOT is within each band)')
acc_bands = [(0, 50, 'Bottom 50% acc'), (25, 75, 'Middle 50% acc'), (50, 100, 'Top 50% acc')]
for lo, hi, label in acc_bands:
    lo_thresh = np.percentile(valid_accs, lo)
    hi_thresh = np.percentile(valid_accs, hi)
    band_mask = (valid_accs > lo_thresh) & (valid_accs <= hi_thresh)
    if band_mask.sum() == 0:
        continue
    band_rd = np.abs(rank_diff[band_mask])
    band_tau, _ = stats.kendalltau(valid_accs[band_mask], valid_scores[band_mask])
    print(f'  {label} ({lo_thresh:.1f}-{hi_thresh:.1f}%): '
          f'n={band_mask.sum():5d}  mean|rank_diff|={band_rd.mean():.0f}  '
          f'tau={band_tau:.4f}')

# ---------------------------------------------------------------------------
# Save failure sets
# ---------------------------------------------------------------------------
os.makedirs(args.save_dir, exist_ok=True)

np.save(os.path.join(args.save_dir, 'false_positive_uids.npy'), fp_uids)
np.save(os.path.join(args.save_dir, 'false_negative_uids.npy'), fn_uids)
np.save(os.path.join(args.save_dir, 'true_positive_uids.npy'), tp_uids)
np.save(os.path.join(args.save_dir, 'true_negative_uids.npy'), tn_uids)
np.save(os.path.join(args.save_dir, 'overrated_uids.npy'), overrated_uids)
np.save(os.path.join(args.save_dir, 'underrated_uids.npy'), underrated_uids)
np.save(os.path.join(args.save_dir, 'all_valid_scores.npy'), valid_scores)
np.save(os.path.join(args.save_dir, 'all_valid_accs.npy'), valid_accs)
np.save(os.path.join(args.save_dir, 'all_valid_uids.npy'), valid_uids)
np.save(os.path.join(args.save_dir, 'rank_diffs.npy'), rank_diff)

print(f'\nFailure sets saved to {args.save_dir}/')
print(f'  false_positive_uids.npy  ({len(fp_uids)} UIDs)')
print(f'  false_negative_uids.npy  ({len(fn_uids)} UIDs)')
print(f'  true_positive_uids.npy   ({len(tp_uids)} UIDs)')
print(f'  true_negative_uids.npy   ({len(tn_uids)} UIDs)')
print(f'  overrated_uids.npy       ({len(overrated_uids)} UIDs -- NASWOT overestimates)')
print(f'  underrated_uids.npy      ({len(underrated_uids)} UIDs -- NASWOT underestimates)')
print(f'  rank_diffs.npy           ({n_valid} entries -- score_rank - acc_rank)')
print(f'  all_valid_*.npy          ({n_valid} entries)')
