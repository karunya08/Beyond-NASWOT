"""
test_feature_diversity.py
=========================
Prototype: measure feature diversity (effective rank of channel correlation
matrix G) for NAS-Bench-201 architectures and check if it correlates with
accuracy.

Idea: NASWOT builds K (samples x samples) from activation masks.
      This script builds G (channels x channels) from actual feature maps.
      G captures: are the network's channels computing different things?

      conv3x3 random filters -> diverse spatial patterns -> high rank G
      conv1x1 random filters -> random channel mixing -> lower diversity

Run from project root:
    python experiments/test_feature_diversity.py --trainval --max_archs 200 --GPU 0

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
from scipy import stats

parser = argparse.ArgumentParser()
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


def effective_rank(M):
    """Effective rank of a matrix via eigenvalue entropy."""
    # M: (channels, spatial) or correlation matrix
    # Compute SVD of M to get singular values
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return np.exp(entropy)


def feature_diversity_score(feature_maps_list):
    """Compute feature diversity from collected feature maps.

    For each layer's feature maps (batch, channels, H, W):
    - Average over batch to get mean activation per channel: (channels, H, W)
    - Flatten each channel: (channels, H*W)
    - Compute effective rank of this matrix

    Returns: sum of effective ranks across layers (like logdet sums across layers).
    """
    total_eff_rank = 0.0
    n_layers = 0
    for fmap in feature_maps_list:
        # fmap: (batch, channels, H, W)
        # Average across batch to get typical channel activations
        mean_fmap = fmap.mean(dim=0)  # (channels, H, W)
        C = mean_fmap.shape[0]
        flat = mean_fmap.reshape(C, -1).cpu().numpy()  # (channels, H*W)

        # Skip if degenerate
        if flat.shape[0] < 2 or flat.shape[1] < 2:
            continue

        er = effective_rank(flat)
        total_eff_rank += er
        n_layers += 1

    return total_eff_rank, n_layers


# Get one batch of data
data_iterator = iter(train_loader)
x_batch, target_batch = next(data_iterator)
x_batch = x_batch.to(device)

n_archs = len(searchspace)
max_archs = min(args.max_archs, n_archs) if args.max_archs > 0 else n_archs

scores_div = np.zeros(max_archs)
scores_logdet = np.zeros(max_archs)
accs = np.zeros(max_archs)
n_hooks_arr = np.zeros(max_archs, dtype=int)

print(f'Scoring {max_archs} architectures...')

for i, (uid, network) in enumerate(searchspace):
    if i >= max_archs:
        break

    try:
        # Collect feature maps from conv layers
        feature_maps = []

        def hook_fn(module, inp, out):
            # out: (batch, channels, H, W)
            if out.dim() == 4 and out.shape[1] > 1:
                feature_maps.append(out.detach())

        hooks = []
        for name, module in network.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn))

        # Also compute standard NASWOT logdet for comparison
        network.K = np.zeros((args.batch_size, args.batch_size))
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                network.K = network.K + K.detach().cpu().numpy() + K2.detach().cpu().numpy()
            except:
                pass

        def counting_backward_hook(module, grad_input, grad_output):
            module.visited_backwards = True

        n_relu_hooks = 0
        for m in network.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                module.register_forward_hook(counting_forward_hook)
                module.register_full_backward_hook(counting_backward_hook)
                n_relu_hooks += 1

        network = network.to(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Backward pass first (for NASWOT hooks)
        x_copy = x_batch.clone().requires_grad_(True)
        y, out = network(x_copy)
        y.backward(torch.ones_like(y))

        # Forward pass (triggers both feature hooks and NASWOT hooks)
        feature_maps.clear()
        with torch.no_grad():
            network(x_batch)

        # Feature diversity score
        div_score, n_div_layers = feature_diversity_score(feature_maps)

        # NASWOT logdet
        s, ld = np.linalg.slogdet(network.K)
        logdet = ld if s > 0 else -np.inf

        scores_div[i] = div_score
        scores_logdet[i] = logdet
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        n_hooks_arr[i] = n_relu_hooks

        # Running tau
        valid = (scores_div[:i+1] > 0) & (accs[:i+1] > 0) & np.isfinite(scores_logdet[:i+1])
        if valid.sum() > 2:
            tau_div, _ = stats.kendalltau(accs[:i+1][valid], scores_div[:i+1][valid])
            tau_ld, _ = stats.kendalltau(accs[:i+1][valid], scores_logdet[:i+1][valid])
        else:
            tau_div = tau_ld = float('nan')

        print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} | feat_div={div_score:>8.2f} | '
              f'logdet={logdet:>8.2f} | acc={accs[i]:>6.2f} | '
              f'tau_div={tau_div:>6.4f} | tau_ld={tau_ld:>6.4f}')

        # Clean up hooks
        for h in hooks:
            h.remove()

    except Exception as e:
        print(f'[{i+1:>3}/{max_archs}] uid={uid:>5} | ERROR: {e}')
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        scores_div[i] = 0
        scores_logdet[i] = float('nan')

# Final analysis
print('\n' + '='*70)
print('FINAL RESULTS')
print('='*70)

valid = (scores_div > 0) & (accs > 0) & np.isfinite(scores_logdet)
s_div = scores_div[valid]
s_ld = scores_logdet[valid]
a = accs[valid]
n_h = n_hooks_arr[valid]

tau_div, _ = stats.kendalltau(a, s_div)
tau_ld, _ = stats.kendalltau(a, s_ld)

print(f'Valid architectures: {valid.sum()}')
print(f'Kendall tau (feature_diversity, acc):           {tau_div:.4f}')
print(f'Kendall tau (logdet, acc):                      {tau_ld:.4f}')

# Correlation between feature diversity and logdet
tau_div_ld, _ = stats.kendalltau(s_div, s_ld)
print(f'Kendall tau (feature_diversity, logdet):        {tau_div_ld:.4f}')

# Correlation with hook count (complexity confound check)
tau_div_hooks, _ = stats.kendalltau(s_div, n_h)
tau_ld_hooks, _ = stats.kendalltau(s_ld, n_h)
print(f'Kendall tau (feature_diversity, n_hooks):       {tau_div_hooks:.4f}')
print(f'Kendall tau (logdet, n_hooks):                  {tau_ld_hooks:.4f}')

# Per-band tau
print('\nPer-band tau:')
for label, lo, hi in [('Bottom 50%', 0, 50), ('Top 50%', 50, 100)]:
    lo_val = np.percentile(a, lo)
    hi_val = np.percentile(a, hi)
    mask = (a > lo_val) & (a <= hi_val) if lo > 0 else (a <= hi_val)
    if mask.sum() > 5:
        t_div, _ = stats.kendalltau(a[mask], s_div[mask])
        t_ld, _ = stats.kendalltau(a[mask], s_ld[mask])
        print(f'  {label} (n={mask.sum()}): '
              f'tau_div={t_div:.4f}, tau_ld={t_ld:.4f}')

# Save results
os.makedirs(args.save_loc, exist_ok=True)
np.save(f'{args.save_loc}/feature_diversity_scores_{max_archs}.npy', scores_div)
np.save(f'{args.save_loc}/feature_diversity_logdet_{max_archs}.npy', scores_logdet)
np.save(f'{args.save_loc}/feature_diversity_accs_{max_archs}.npy', accs)
np.save(f'{args.save_loc}/feature_diversity_nhooks_{max_archs}.npy', n_hooks_arr)
print(f'\nResults saved to {args.save_loc}/')
