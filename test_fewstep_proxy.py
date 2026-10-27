"""
test_fewstep_proxy.py
=====================
Can a few training steps reveal what init-time metrics miss about skip
connections?

Background:
  - NASWOT (init-time) has tau=0.57 but is blind to skip connections
  - NTK, grad/weight, 1-step loss, eigenNAS: all fail (see approaches 1-11)
  - Root cause: skip connections improve training dynamics OVER TIME,
    not init-time statistics. Kaiming init equalizes everything at init.

New approaches tested here:
  A) Few-step training proxy: train K steps, measure VALIDATION loss
     -> Direct trainability measurement. Unlike 1-step proxy (tau=-0.17),
        uses DIFFERENT batches for train vs eval, and K>1 steps.
  B) NASWOT after K training steps: measure activation diversity change
     -> Hypothesis: overrated archs LOSE activation diversity after a few
        steps (ineffective training), skip archs MAINTAIN it.
     -> naswot_delta = logdet_after - logdet_init captures training dynamics
  C) Gradient consistency: cos(grad_batch1, grad_batch2) at init
     -> Skip connections provide a data-independent gradient path (identity),
        so gradients should be more consistent across different data batches.

All metrics stratified by skip connection topology.

Usage:
    python test_fewstep_proxy.py --trainval --max_archs 200 --GPU 0
    python test_fewstep_proxy.py --trainval --max_archs 200 --GPU 0 --lr 0.1
"""

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import copy
import torch
import torch.nn as nn
from scipy import stats
from models import get_cell_based_tiny_net

parser = argparse.ArgumentParser(description='Few-step trainability proxy')
parser.add_argument('--data_loc', default='../cifardata/', type=str)
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--nasspace', default='nasbench201', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--max_archs', default=200, type=int)
parser.add_argument('--save_loc', default='results', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='SGD learning rate')
parser.add_argument('--k_max', default=20, type=int, help='max training steps')
parser.add_argument('--epsilon', default=0.1, type=float, help='perturbation scale')
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
savedataset = args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval,
                                  args.batch_size, 'none', 1, args)

acc_type = 'ori-test' if args.dataset == 'cifar10' else 'x-test'


# ============================================================
# Pre-load data batches (fixed across all architectures)
# ============================================================
print('Pre-loading data batches...')
data_iter = iter(train_loader)
all_batches = []
for _ in range(args.k_max + 5):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)
    all_batches.append((x.to(device), y.to(device)))

naswot_x = all_batches[0][0]           # for NASWOT (x only, 128 samples)
val_x, val_y = all_batches[1]          # validation batch
grad_x1, grad_y1 = all_batches[2]      # gradient consistency batch 1
grad_x2, grad_y2 = all_batches[3]      # gradient consistency batch 2
train_batches = all_batches[4:]         # training batches (k_max available)
print(f'  Loaded {len(all_batches)} batches: 1 NASWOT, 1 val, 2 grad, {len(train_batches)} train')


# ============================================================
# Helpers
# ============================================================
def get_network_10class(uid, seed=None):
    """Build NAS-Bench-201 network with real num_classes for cross-entropy."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    config = searchspace.api.get_net_config(uid, 'cifar10-valid')
    config['num_classes'] = n_classes
    net = get_cell_based_tiny_net(config)
    for m in net.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False
    return net.to(device)


def compute_naswot_logdet(network, x_batch):
    """Compute NASWOT logdet with self-contained hook management."""
    N = x_batch.shape[0]
    K = np.zeros((N, N))
    hooks = []

    def fwd_hook(module, inp, out):
        nonlocal K
        try:
            if not module._naswot_visited:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K += (x @ x.t() + (1. - x) @ (1. - x.t())).cpu().numpy()
        except:
            pass

    def bwd_hook(module, gi, go):
        module._naswot_visited = True

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            module._naswot_visited = False
            hooks.append(module.register_forward_hook(fwd_hook))
            hooks.append(module.register_full_backward_hook(bwd_hook))

    was_training = network.training
    network.train()
    network.zero_grad()
    x_copy = x_batch.clone().requires_grad_(True)
    y, out = network(x_copy)
    y.backward(torch.ones_like(y))

    with torch.no_grad():
        network(x_batch)

    for h in hooks:
        h.remove()
    network.zero_grad()
    network.train(was_training)

    s, ld = np.linalg.slogdet(K)
    return ld if s > 0 else -np.inf


def compute_grad_consistency(network, x1, y1, x2, y2):
    """Cosine similarity of parameter gradients on two different batches.

    Skip connections provide a data-independent gradient path (identity),
    so gradients should be more consistent across batches.
    """
    network.train()
    criterion = nn.CrossEntropyLoss()

    network.zero_grad()
    logits1, _ = network(x1)
    criterion(logits1, y1).backward()
    g1 = torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None])

    network.zero_grad()
    logits2, _ = network(x2)
    criterion(logits2, y2).backward()
    g2 = torch.cat([p.grad.flatten() for p in network.parameters() if p.grad is not None])

    network.zero_grad()
    return (g1 @ g2).item() / (g1.norm().item() * g2.norm().item() + 1e-10)


def get_topology(api, uid):
    """Parse architecture and return skip/topology features."""
    config = api.get_net_config(uid, 'cifar10-valid')
    nodes = api.str2lists(config['arch_str'])
    ops = [op for node_inputs in nodes for (op, src) in node_inputs]

    n_skip = sum(1 for op in ops if op == 'skip_connect')
    n_parametric = sum(1 for op in ops if op in ('nor_conv_1x1', 'nor_conv_3x3'))
    n_conv3x3 = sum(1 for op in ops if op == 'nor_conv_3x3')

    # True residual: skip 0->3 AND active conv path to node 3
    has_true_res = 0
    if ops[3] == 'skip_connect':
        paths = {'via_1': [ops[0], ops[4]], 'via_2': [ops[1], ops[5]],
                 'via_1_2': [ops[0], ops[2], ops[5]]}
        for path_ops in paths.values():
            if (all(op != 'none' for op in path_ops) and
                    any(op in ('nor_conv_1x1', 'nor_conv_3x3') for op in path_ops)):
                has_true_res = 1
                break

    return n_skip, n_parametric, n_conv3x3, has_true_res


# ============================================================
# Main scoring loop
# ============================================================
max_archs = min(args.max_archs, len(searchspace))
CHECKPOINTS = [k for k in [5, 10, 20] if k <= args.k_max]

# Metric names
INIT_METRICS = ['logdet_init', 'grad_consistency']
STEP_METRICS = []
for k in CHECKPOINTS:
    STEP_METRICS += [f'neg_val_loss_{k}', f'val_acc_{k}',
                     f'logdet_after_{k}', f'naswot_delta_{k}']
ALL_METRICS = INIT_METRICS + STEP_METRICS

all_scores = {m: np.full(max_archs, np.nan) for m in ALL_METRICS}
accs = np.zeros(max_archs)
n_skips = np.zeros(max_archs, dtype=int)
n_params = np.zeros(max_archs, dtype=int)
has_true_res = np.zeros(max_archs, dtype=int)

criterion = nn.CrossEntropyLoss()

print(f'\nScoring {max_archs} architectures (K_max={args.k_max}, lr={args.lr})')
print(f'Checkpoints at steps: {CHECKPOINTS}')
print()

for i, (uid, _) in enumerate(searchspace):
    if i >= max_archs:
        break

    try:
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        n_skip, n_param, n_c3, true_res = get_topology(searchspace.api, uid)
        n_skips[i] = n_skip
        n_params[i] = n_param
        has_true_res[i] = true_res

        # Deterministic init per architecture
        init_seed = args.seed * 10000 + uid
        net = get_network_10class(uid, seed=init_seed)

        # ---- A: NASWOT at init ----
        logdet_init = compute_naswot_logdet(net, naswot_x)
        all_scores['logdet_init'][i] = logdet_init

        # ---- B: Gradient consistency ----
        grad_cos = compute_grad_consistency(net, grad_x1, grad_y1, grad_x2, grad_y2)
        all_scores['grad_consistency'][i] = grad_cos

        # ---- C: Few-step training ----
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)
        net.train()

        for step in range(args.k_max):
            tx, ty = train_batches[step % len(train_batches)]
            optimizer.zero_grad()
            logits, _ = net(tx)
            loss = criterion(logits, ty)
            loss.backward()
            optimizer.step()

            if (step + 1) in CHECKPOINTS:
                k = step + 1
                # Validation on held-out batch
                with torch.no_grad():
                    vl, _ = net(val_x)
                    vloss = criterion(vl, val_y).item()
                    vacc = (vl.argmax(1) == val_y).float().mean().item()
                all_scores[f'neg_val_loss_{k}'][i] = -vloss  # negate: higher=better
                all_scores[f'val_acc_{k}'][i] = vacc

                # NASWOT after training (activation diversity change)
                logdet_after = compute_naswot_logdet(net, naswot_x)
                all_scores[f'logdet_after_{k}'][i] = logdet_after
                all_scores[f'naswot_delta_{k}'][i] = logdet_after - logdet_init

        del net
        torch.cuda.empty_cache()

        # Running tau
        valid_so_far = (accs[:i+1] > 0)
        for m in ALL_METRICS:
            valid_so_far = valid_so_far & np.isfinite(all_scores[m][:i+1])
        n_valid = valid_so_far.sum()
        if n_valid > 5:
            tau_ld, _ = stats.kendalltau(accs[:i+1][valid_so_far],
                                          all_scores['logdet_init'][:i+1][valid_so_far])
            tau_gc, _ = stats.kendalltau(accs[:i+1][valid_so_far],
                                          all_scores['grad_consistency'][:i+1][valid_so_far])
            k_last = CHECKPOINTS[-1]
            tau_va, _ = stats.kendalltau(accs[:i+1][valid_so_far],
                                          all_scores[f'val_acc_{k_last}'][:i+1][valid_so_far])
            tau_nd, _ = stats.kendalltau(accs[:i+1][valid_so_far],
                                          all_scores[f'naswot_delta_{k_last}'][:i+1][valid_so_far])
            print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} acc={accs[i]:>5.1f} '
                  f'skip={n_skip} res={true_res} | '
                  f'logdet={tau_ld:.3f} gc={tau_gc:.3f} '
                  f'vacc{k_last}={tau_va:.3f} delta{k_last}={tau_nd:.3f}')
        else:
            print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} acc={accs[i]:>5.1f} skip={n_skip}')

    except Exception as e:
        print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} ERROR: {e}')
        import traceback; traceback.print_exc()
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)


# ============================================================
# FINAL RESULTS
# ============================================================
print(f'\n{"="*90}')
print(f'FINAL RESULTS  ({max_archs} architectures, lr={args.lr}, K_max={args.k_max})')
print(f'{"="*90}')

valid = (accs > 0)
for m in ALL_METRICS:
    valid = valid & np.isfinite(all_scores[m])
a = accs[valid]
print(f'Valid: {valid.sum()}\n')

# Overall tau
print('OVERALL KENDALL TAU')
print(f'  {"Metric":<25s} {"tau":>8s}')
print('  ' + '-' * 35)
taus = {}
for m in ALL_METRICS:
    t, _ = stats.kendalltau(a, all_scores[m][valid])
    taus[m] = t
    print(f'  {m:<25s} {t:>8.4f}')


# ============================================================
# Stratified analysis
# ============================================================
def stratified_table(title, mask_a, label_a, mask_b, label_b):
    """Print tau table for two groups."""
    ga = mask_a & valid
    gb = mask_b & valid
    if ga.sum() < 5 or gb.sum() < 5:
        print(f'\n  {title}: insufficient data (A={ga.sum()}, B={gb.sum()})')
        return

    print(f'\n{"="*90}')
    print(title)
    print(f'{"="*90}')
    print(f'  {label_a}: n={ga.sum()}, mean_acc={accs[ga].mean():.2f}%')
    print(f'  {label_b}: n={gb.sum()}, mean_acc={accs[gb].mean():.2f}%')

    print(f'\n  {"Metric":<25s} {"Overall":>8s} {label_a:>10s} {label_b:>10s} {"Diff":>8s}')
    print('  ' + '-' * 65)

    for m in ALL_METRICS:
        t_all = taus[m]
        t_a, _ = stats.kendalltau(accs[ga], all_scores[m][ga])
        t_b, _ = stats.kendalltau(accs[gb], all_scores[m][gb])
        diff = t_a - t_b
        print(f'  {m:<25s} {t_all:>8.4f} {t_a:>10.4f} {t_b:>10.4f} {diff:>+8.4f}')

    # Mean scores per group
    print(f'\n  Mean scores:')
    print(f'  {"Metric":<25s} {label_a:>12s} {label_b:>12s} {"Diff":>10s}')
    print('  ' + '-' * 62)
    for m in ALL_METRICS:
        va = all_scores[m][ga]; va = va[np.isfinite(va)]
        vb = all_scores[m][gb]; vb = vb[np.isfinite(vb)]
        ma, mb = va.mean(), vb.mean()
        print(f'  {m:<25s} {ma:>12.4f} {mb:>12.4f} {ma - mb:>+10.4f}')


# Skip vs no-skip
stratified_table('SKIP (>=1) vs NO SKIP (0)',
                 n_skips >= 1, 'Skip>=1', n_skips == 0, 'Skip=0')

# True residual
stratified_table('TRUE RESIDUAL vs NO TRUE RESIDUAL',
                 has_true_res == 1, 'TrueRes', has_true_res == 0, 'NoTrueRes')


# ============================================================
# Per skip-count breakdown
# ============================================================
print(f'\n{"="*90}')
print('TAU BY SKIP COUNT')
print(f'{"="*90}')

key_metrics = ['logdet_init', 'grad_consistency'] + \
              [f'val_acc_{k}' for k in CHECKPOINTS] + \
              [f'naswot_delta_{k}' for k in CHECKPOINTS]

for ns in sorted(set(n_skips)):
    mask = (n_skips == ns) & valid
    if mask.sum() < 5:
        continue
    print(f'\n  n_skip={ns}  (n={mask.sum()}, mean_acc={accs[mask].mean():.2f}%):')
    for m in key_metrics:
        if m in ALL_METRICS:
            t, _ = stats.kendalltau(accs[mask], all_scores[m][mask])
            print(f'    {m:<25s}: tau={t:.4f}  mean={all_scores[m][mask].mean():.4f}')


# ============================================================
# KEY: Which metric correctly ranks skip vs non-skip?
# ============================================================
print(f'\n{"="*90}')
print('KEY QUESTION: Which metric correctly values skip connections?')
print(f'{"="*90}')

has_s = (n_skips >= 1) & valid
no_s = (n_skips == 0) & valid
if has_s.sum() > 5 and no_s.sum() > 5:
    skip_acc = accs[has_s].mean()
    noskip_acc = accs[no_s].mean()
    better = 'SKIP' if skip_acc > noskip_acc else 'NO-SKIP'
    print(f'\n  Ground truth: Skip={skip_acc:.2f}%, NoSkip={noskip_acc:.2f}%  '
          f'({better} is better by {abs(skip_acc - noskip_acc):.2f}%)')

    print(f'\n  {"Metric":<25s} {"Skip mean":>10s} {"NoSkip mean":>10s} '
          f'{"Higher for":>12s} {"Correct?":>10s}')
    print('  ' + '-' * 72)
    for m in ALL_METRICS:
        skip_m = all_scores[m][has_s]; skip_m = skip_m[np.isfinite(skip_m)].mean()
        nosk_m = all_scores[m][no_s]; nosk_m = nosk_m[np.isfinite(nosk_m)].mean()
        higher_for = 'Skip' if skip_m > nosk_m else 'NoSkip'
        correct = (skip_m > nosk_m) == (skip_acc > noskip_acc)
        print(f'  {m:<25s} {skip_m:>10.4f} {nosk_m:>10.4f} '
              f'{higher_for:>12s} {"YES" if correct else "NO":>10s}')

# Same for true residual
has_tr = (has_true_res == 1) & valid
no_tr = (has_true_res == 0) & valid
if has_tr.sum() > 5 and no_tr.sum() > 5:
    tr_acc = accs[has_tr].mean()
    notr_acc = accs[no_tr].mean()
    print(f'\n  True residual: TrueRes={tr_acc:.2f}%, NoTrueRes={notr_acc:.2f}%')
    print(f'\n  {"Metric":<25s} {"TrueRes":>10s} {"NoTrueRes":>10s} {"Correct?":>10s}')
    print('  ' + '-' * 58)
    for m in ALL_METRICS:
        tr_m = all_scores[m][has_tr]; tr_m = tr_m[np.isfinite(tr_m)].mean()
        notr_m = all_scores[m][no_tr]; notr_m = notr_m[np.isfinite(notr_m)].mean()
        correct = (tr_m > notr_m) == (tr_acc > notr_acc)
        print(f'  {m:<25s} {tr_m:>10.4f} {notr_m:>10.4f} {"YES" if correct else "NO":>10s}')


# ============================================================
# Combined scores: NASWOT + new metric
# ============================================================
print(f'\n{"="*90}')
print('COMBINED SCORES (logdet_init + alpha * new_metric)')
print(f'{"="*90}')

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

ld_n = normalize(all_scores['logdet_init'][valid])

for combo_metric in ['grad_consistency'] + \
                     [f'val_acc_{k}' for k in CHECKPOINTS] + \
                     [f'naswot_delta_{k}' for k in CHECKPOINTS] + \
                     [f'neg_val_loss_{k}' for k in CHECKPOINTS]:
    if combo_metric not in ALL_METRICS:
        continue
    cm_n = normalize(all_scores[combo_metric][valid])
    print(f'\n  logdet + alpha * {combo_metric}:')
    for alpha in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        combined = ld_n + alpha * cm_n
        t, _ = stats.kendalltau(a, combined)
        marker = ' <--' if t > taus['logdet_init'] + 0.01 else ''
        print(f'    alpha={alpha:<5.1f}: tau={t:.4f}{marker}')

# Pure new metrics (no logdet)
print(f'\n  Standalone new metrics vs logdet baseline (tau={taus["logdet_init"]:.4f}):')
for m in ALL_METRICS:
    if m != 'logdet_init':
        t = taus[m]
        diff = t - taus['logdet_init']
        print(f'    {m:<25s}: tau={t:.4f}  ({diff:+.4f} vs logdet)')


# ============================================================
# Overrated vs Underrated (if failure_sets available)
# ============================================================
failure_dir = 'failure_sets'
if os.path.exists(os.path.join(failure_dir, 'overrated_uids.npy')):
    print(f'\n{"="*90}')
    print('OVERRATED vs UNDERRATED')
    print(f'{"="*90}')

    overrated_uids = set(np.load(os.path.join(failure_dir, 'overrated_uids.npy')).tolist())
    underrated_uids = set(np.load(os.path.join(failure_dir, 'underrated_uids.npy')).tolist())

    over_mask = np.array([i in overrated_uids for i in range(max_archs)]) & valid
    under_mask = np.array([i in underrated_uids for i in range(max_archs)]) & valid

    if over_mask.sum() > 2 and under_mask.sum() > 2:
        print(f'\n  Overrated  (n={over_mask.sum()}): '
              f'mean_acc={accs[over_mask].mean():.2f}%, mean_skip={n_skips[over_mask].mean():.1f}')
        print(f'  Underrated (n={under_mask.sum()}): '
              f'mean_acc={accs[under_mask].mean():.2f}%, mean_skip={n_skips[under_mask].mean():.1f}')

        print(f'\n  {"Metric":<25s} {"Overrated":>10s} {"Underrated":>10s} '
              f'{"Under-Over":>12s} {"Correct dir?":>14s}')
        print('  ' + '-' * 76)
        for m in ALL_METRICS:
            ov = all_scores[m][over_mask]; ov = ov[np.isfinite(ov)]
            un = all_scores[m][under_mask]; un = un[np.isfinite(un)]
            if len(ov) == 0 or len(un) == 0:
                continue
            ovm, unm = ov.mean(), un.mean()
            # "Correct direction" = underrated scores higher (since they're actually better)
            correct = unm > ovm
            print(f'  {m:<25s} {ovm:>10.4f} {unm:>10.4f} '
                  f'{unm - ovm:>+12.4f} {"YES" if correct else "NO":>14s}')


# ============================================================
# Save results
# ============================================================
os.makedirs(args.save_loc, exist_ok=True)
prefix = f'{args.save_loc}/fewstep_{max_archs}_lr{args.lr}'
for m in ALL_METRICS:
    np.save(f'{prefix}_{m}.npy', all_scores[m])
np.save(f'{prefix}_accs.npy', accs)
np.save(f'{prefix}_nskips.npy', n_skips)
np.save(f'{prefix}_nparams.npy', n_params)
np.save(f'{prefix}_true_res.npy', has_true_res)
print(f'\nResults saved to {prefix}_*.npy')
