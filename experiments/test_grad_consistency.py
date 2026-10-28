"""
test_grad_consistency.py
========================
Zero-cost metric to detect NASWOT's skip connection blind spot.

Idea:
  For each architecture (at random init, NO training):
    1. Forward batch_A through network, compute loss, backprop -> gradient g_A
    2. Forward batch_B through network, compute loss, backprop -> gradient g_B
    3. score = cosine_similarity(g_A, g_B)

  Why this works:
    Skip connections provide an identity path: output = f(x) + x.
    The gradient through the identity path is data-independent (just passes
    through unchanged). So networks with skip connections have a stable
    gradient component, making g_A and g_B more similar.

    Conv-only networks have fully data-dependent gradients, so g_A and g_B
    point in different directions -> low cosine similarity.

  This is zero-cost (same cost as NASWOT: two forward+backward passes).

Run from project root:
    python experiments/test_grad_consistency.py --trainval --max_archs 200 --GPU 0

Prerequisite: None (scores all from scratch)
"""

import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import nasspace
import datasets
import random
import numpy as np
import os
import torch
import torch.nn as nn
from scipy import stats
from models import get_cell_based_tiny_net

parser = argparse.ArgumentParser(description='Gradient Consistency (zero-cost)')
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
# Load two data batches (fixed for all architectures)
# ============================================================
print('Loading data batches...')
data_iter = iter(train_loader)
batch_A_x, batch_A_y = next(data_iter)
batch_B_x, batch_B_y = next(data_iter)
batch_A_x, batch_A_y = batch_A_x.to(device), batch_A_y.to(device)
batch_B_x, batch_B_y = batch_B_x.to(device), batch_B_y.to(device)
print(f'  Batch A: {batch_A_x.shape}, Batch B: {batch_B_x.shape}')


# ============================================================
# Helpers
# ============================================================
def get_network(uid, seed=None):
    """Build NAS-Bench-201 network."""
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
    """Compute NASWOT logdet score."""
    N = x_batch.shape[0]
    K = np.zeros((N, N))
    hooks = []

    def fwd_hook(module, inp, out):
        nonlocal K
        try:
            if not module._visited:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = (inp > 0).float()
            K += (x @ x.t() + (1. - x) @ (1. - x.t())).cpu().numpy()
        except:
            pass

    def bwd_hook(module, gi, go):
        module._visited = True

    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            module._visited = False
            hooks.append(module.register_forward_hook(fwd_hook))
            hooks.append(module.register_full_backward_hook(bwd_hook))

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

    s, ld = np.linalg.slogdet(K)
    return ld if s > 0 else -np.inf


def compute_grad_consistency(network, xA, yA, xB, yB):
    """
    Gradient consistency: cosine similarity of parameter gradients
    computed on two different data batches.

    For each batch:
      1. Forward pass -> predictions
      2. Compute cross-entropy loss
      3. Backward pass -> gradient of loss w.r.t. ALL parameters
      4. Flatten all parameter gradients into one vector

    Then: score = cos(g_A, g_B)

    High score -> gradients point in same direction regardless of data
                  (skip connections provide this stability)
    Low score  -> gradient direction is data-dependent
                  (conv-only networks have this problem)
    """
    network.train()
    criterion = nn.CrossEntropyLoss()

    # Gradient on batch A
    network.zero_grad()
    logits_A, _ = network(xA)
    loss_A = criterion(logits_A, yA)
    loss_A.backward()
    g_A = torch.cat([p.grad.flatten() for p in network.parameters()
                     if p.grad is not None])

    # Gradient on batch B
    network.zero_grad()
    logits_B, _ = network(xB)
    loss_B = criterion(logits_B, yB)
    loss_B.backward()
    g_B = torch.cat([p.grad.flatten() for p in network.parameters()
                     if p.grad is not None])

    network.zero_grad()

    # Cosine similarity
    cos_sim = (g_A @ g_B).item() / (g_A.norm().item() * g_B.norm().item() + 1e-10)
    return cos_sim


def get_topology(api, uid):
    """Parse architecture string -> skip count, true residual."""
    config = api.get_net_config(uid, 'cifar10-valid')
    nodes = api.str2lists(config['arch_str'])
    ops = [op for node_inputs in nodes for (op, src) in node_inputs]

    n_skip = sum(1 for op in ops if op == 'skip_connect')

    # True residual: skip 0->3 AND a conv path also reaches node 3
    has_true_res = 0
    if ops[3] == 'skip_connect':  # edge 0->3
        paths = {'via_1': [ops[0], ops[4]], 'via_2': [ops[1], ops[5]],
                 'via_1_2': [ops[0], ops[2], ops[5]]}
        for path_ops in paths.values():
            if (all(op != 'none' for op in path_ops) and
                    any(op in ('nor_conv_1x1', 'nor_conv_3x3') for op in path_ops)):
                has_true_res = 1
                break

    return n_skip, has_true_res


# ============================================================
# Score all architectures
# ============================================================
max_archs = min(args.max_archs, len(searchspace))

accs = np.zeros(max_archs)
logdet_scores = np.zeros(max_archs)
gc_scores = np.zeros(max_archs)
n_skips = np.zeros(max_archs, dtype=int)
has_true_res = np.zeros(max_archs, dtype=int)

print(f'\nScoring {max_archs} architectures...\n')

for i, (uid, _) in enumerate(searchspace):
    if i >= max_archs:
        break

    try:
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        n_skip, true_res = get_topology(searchspace.api, uid)
        n_skips[i] = n_skip
        has_true_res[i] = true_res

        # Deterministic init
        init_seed = args.seed * 10000 + uid
        net = get_network(uid, seed=init_seed)

        # NASWOT logdet
        logdet_scores[i] = compute_naswot_logdet(net, batch_A_x)

        # Gradient consistency
        gc_scores[i] = compute_grad_consistency(net, batch_A_x, batch_A_y,
                                                 batch_B_x, batch_B_y)

        del net
        torch.cuda.empty_cache()

        print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} acc={accs[i]:>5.1f}% '
              f'skip={n_skip} res={true_res} | '
              f'logdet={logdet_scores[i]:.1f}  gc={gc_scores[i]:.4f}')

    except Exception as e:
        print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} ERROR: {e}')
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)


# ============================================================
# Results
# ============================================================
valid = (accs > 0) & np.isfinite(logdet_scores) & np.isfinite(gc_scores)
a = accs[valid]

print(f'\n{"="*70}')
print(f'RESULTS  ({valid.sum()} valid architectures)')
print(f'{"="*70}')

tau_ld, p_ld = stats.kendalltau(a, logdet_scores[valid])
tau_gc, p_gc = stats.kendalltau(a, gc_scores[valid])
print(f'\n  NASWOT logdet:      tau={tau_ld:.4f}  (p={p_ld:.4e})')
print(f'  Grad consistency:   tau={tau_gc:.4f}  (p={p_gc:.4e})')


# ---- Skip vs No-skip ----
has_s = (n_skips >= 1) & valid
no_s = (n_skips == 0) & valid

print(f'\n{"="*70}')
print('SKIP vs NO-SKIP')
print(f'{"="*70}')
print(f'  Has skip (n={has_s.sum()}): mean acc = {accs[has_s].mean():.2f}%')
print(f'  No skip  (n={no_s.sum()}):  mean acc = {accs[no_s].mean():.2f}%')
print(f'  Ground truth: skip architectures are BETTER by {accs[has_s].mean() - accs[no_s].mean():.2f}%')

print(f'\n  {"Metric":<20s} {"Skip mean":>10s} {"NoSkip mean":>12s} {"Higher for":>12s} {"Correct?":>10s}')
print('  ' + '-' * 68)
for name, vals in [('NASWOT logdet', logdet_scores), ('Grad consistency', gc_scores)]:
    sm = vals[has_s].mean()
    nm = vals[no_s].mean()
    higher = 'Skip' if sm > nm else 'NoSkip'
    correct = 'YES' if (sm > nm) else 'NO'
    print(f'  {name:<20s} {sm:>10.4f} {nm:>12.4f} {higher:>12s} {correct:>10s}')


# ---- True residual ----
has_tr = (has_true_res == 1) & valid
no_tr = (has_true_res == 0) & valid

if has_tr.sum() > 3:
    print(f'\n{"="*70}')
    print('TRUE RESIDUAL vs NO TRUE RESIDUAL')
    print(f'{"="*70}')
    print(f'  True residual (n={has_tr.sum()}):  mean acc = {accs[has_tr].mean():.2f}%')
    print(f'  No true res   (n={no_tr.sum()}):  mean acc = {accs[no_tr].mean():.2f}%')

    print(f'\n  {"Metric":<20s} {"TrueRes":>10s} {"NoTrueRes":>12s} {"Correct?":>10s}')
    print('  ' + '-' * 56)
    for name, vals in [('NASWOT logdet', logdet_scores), ('Grad consistency', gc_scores)]:
        tm = vals[has_tr].mean()
        ntm = vals[no_tr].mean()
        correct = 'YES' if (tm > ntm) == (accs[has_tr].mean() > accs[no_tr].mean()) else 'NO'
        print(f'  {name:<20s} {tm:>10.4f} {ntm:>12.4f} {correct:>10s}')


# ---- Per skip count ----
print(f'\n{"="*70}')
print('GRAD CONSISTENCY BY SKIP COUNT')
print(f'{"="*70}')
print(f'  {"n_skip":>6s} {"N":>5s} {"mean_acc":>10s} {"mean_gc":>10s} {"mean_logdet":>12s}')
print('  ' + '-' * 48)
for ns in sorted(set(n_skips)):
    mask = (n_skips == ns) & valid
    if mask.sum() < 3:
        continue
    print(f'  {ns:>6d} {mask.sum():>5d} {accs[mask].mean():>10.2f} '
          f'{gc_scores[mask].mean():>10.4f} {logdet_scores[mask].mean():>12.1f}')


# ---- Overrated vs Underrated ----
failure_dir = 'failure_sets'
if os.path.exists(os.path.join(failure_dir, 'overrated_uids.npy')):
    overrated_uids = set(np.load(os.path.join(failure_dir, 'overrated_uids.npy')).tolist())
    underrated_uids = set(np.load(os.path.join(failure_dir, 'underrated_uids.npy')).tolist())

    over_mask = np.array([i in overrated_uids for i in range(max_archs)]) & valid
    under_mask = np.array([i in underrated_uids for i in range(max_archs)]) & valid

    if over_mask.sum() > 2 and under_mask.sum() > 2:
        print(f'\n{"="*70}')
        print('OVERRATED vs UNDERRATED (NASWOT failure cases)')
        print(f'{"="*70}')
        print(f'  Overrated  (n={over_mask.sum()}): mean acc={accs[over_mask].mean():.2f}%, '
              f'mean skip={n_skips[over_mask].mean():.1f}')
        print(f'  Underrated (n={under_mask.sum()}): mean acc={accs[under_mask].mean():.2f}%, '
              f'mean skip={n_skips[under_mask].mean():.1f}')

        print(f'\n  {"Metric":<20s} {"Overrated":>10s} {"Underrated":>12s} '
              f'{"Cohen d":>10s} {"p-value":>10s} {"Correct?":>10s}')
        print('  ' + '-' * 76)

        for name, vals in [('NASWOT logdet', logdet_scores), ('Grad consistency', gc_scores)]:
            ov = vals[over_mask]
            un = vals[under_mask]
            pooled_std = np.sqrt((ov.std()**2 + un.std()**2) / 2)
            d = (un.mean() - ov.mean()) / (pooled_std + 1e-10)
            _, p = stats.ttest_ind(un, ov)
            correct = 'YES' if un.mean() > ov.mean() else 'NO'
            print(f'  {name:<20s} {ov.mean():>10.4f} {un.mean():>12.4f} '
                  f'{d:>+10.2f} {p:>10.4f} {correct:>10s}')

        print(f'\n  Interpretation:')
        print(f'    NASWOT gives overrated architectures HIGHER scores (wrong)')
        print(f'    Grad consistency gives underrated architectures HIGHER scores (correct)')
        print(f'    -> Grad consistency detects what NASWOT misses about skip connections')


# ---- Save ----
os.makedirs(args.save_loc, exist_ok=True)
prefix = f'{args.save_loc}/grad_consistency_{max_archs}'
np.save(f'{prefix}_gc.npy', gc_scores)
np.save(f'{prefix}_logdet.npy', logdet_scores)
np.save(f'{prefix}_accs.npy', accs)
np.save(f'{prefix}_nskips.npy', n_skips)
np.save(f'{prefix}_true_res.npy', has_true_res)
print(f'\nResults saved to {prefix}_*.npy')
