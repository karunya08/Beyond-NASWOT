"""
compare_overrated_underrated.py
================================
Compute feature diversity for overrated and underrated architectures
from the failure analysis, then compare distributions.

If the theory is correct:
  - Overrated:   high logdet, LOW feature diversity (many cheap redundant channels)
  - Underrated:  modest logdet, HIGH feature diversity (fewer but diverse channels)

Usage:
    python compare_overrated_underrated.py --GPU 0
"""

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import torch
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', default='../cifardata/', type=str)
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--nasspace', default='nasbench201', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--failure_dir', default='failure_sets', type=str)
parser.add_argument('--save_loc', default='results', type=str)
parser.add_argument('--stem_out_channels', default=16, type=int)
parser.add_argument('--num_stacks', default=3, type=int)
parser.add_argument('--num_modules_per_stack', default=3, type=int)
parser.add_argument('--num_labels', default=1, type=int)
parser.add_argument('--n_sample', default=200, type=int,
                    help='sample this many from each group (0 = all)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load failure sets
overrated_uids = np.load(os.path.join(args.failure_dir, 'overrated_uids.npy'))
underrated_uids = np.load(os.path.join(args.failure_dir, 'underrated_uids.npy'))
print(f'Overrated: {len(overrated_uids)} architectures')
print(f'Underrated: {len(underrated_uids)} architectures')

# Sample if needed
if args.n_sample > 0:
    if len(overrated_uids) > args.n_sample:
        overrated_uids = np.random.choice(overrated_uids, args.n_sample, replace=False)
    if len(underrated_uids) > args.n_sample:
        underrated_uids = np.random.choice(underrated_uids, args.n_sample, replace=False)
    print(f'Sampled: {len(overrated_uids)} overrated, {len(underrated_uids)} underrated')

# Load full baseline scores and accs
scores_file = f'{args.save_loc}/full_baseline_hook_logdet_{args.nasspace}_{args.dataset}__none_0.05_1_True_128_1_1.npy'
accs_file = f'{args.save_loc}/full_baseline_accs_{args.nasspace}_{args.dataset}_True.npy'
all_scores = np.load(scores_file)
all_accs = np.load(accs_file)

searchspace = nasspace.get_search_space(args)
savedataset = args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval,
                                  args.batch_size, 'none', 1, args)

# Get one batch
data_iterator = iter(train_loader)
x_batch, target_batch = next(data_iterator)
x_batch = x_batch.to(device)


def effective_rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-15))
    return np.exp(entropy)


def compute_feature_diversity(network, x, device, seed=1):
    """Compute feature diversity and logdet for a single architecture."""
    feature_maps = []

    def hook_fn(module, inp, out):
        if out.dim() == 4 and out.shape[1] > 1:
            feature_maps.append(out.detach())

    hooks = []
    for name, module in network.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn))

    # NASWOT logdet
    network.K = np.zeros((x.shape[0], x.shape[0]))
    n_relu_hooks = 0

    def counting_forward_hook(module, inp, out):
        try:
            if not module.visited_backwards:
                return
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            xb = (inp > 0).float()
            K = xb @ xb.t()
            K2 = (1. - xb) @ (1. - xb.t())
            network.K = network.K + K.detach().cpu().numpy() + K2.detach().cpu().numpy()
        except:
            pass

    def counting_backward_hook(module, grad_input, grad_output):
        module.visited_backwards = True

    for m in network.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    for name, module in network.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)
            module.register_full_backward_hook(counting_backward_hook)
            n_relu_hooks += 1

    network = network.to(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Backward pass
    x_copy = x.clone().requires_grad_(True)
    y, out = network(x_copy)
    y.backward(torch.ones_like(y))

    # Forward pass
    feature_maps.clear()
    with torch.no_grad():
        network(x)

    # Feature diversity (batch-averaged)
    total_eff_rank = 0.0
    for fmap in feature_maps:
        mean_fmap = fmap.mean(dim=0)
        C = mean_fmap.shape[0]
        flat = mean_fmap.reshape(C, -1).cpu().numpy()
        if flat.shape[0] < 2 or flat.shape[1] < 2:
            continue
        total_eff_rank += effective_rank(flat)

    # Logdet
    s, ld = np.linalg.slogdet(network.K)
    logdet = ld if s > 0 else -np.inf

    for h in hooks:
        h.remove()

    return total_eff_rank, logdet, n_relu_hooks


def score_group(uids, label):
    feat_divs = []
    logdets = []
    n_hooks_list = []
    accuracies = []

    for idx, uid in enumerate(uids):
        try:
            network = searchspace.get_network(int(uid))
            fd, ld, nh = compute_feature_diversity(network, x_batch, device, args.seed)
            feat_divs.append(fd)
            logdets.append(ld)
            n_hooks_list.append(nh)
            accuracies.append(all_accs[int(uid)])

            if (idx + 1) % 50 == 0 or idx == len(uids) - 1:
                print(f'  [{label}] {idx+1}/{len(uids)} done')
        except Exception as e:
            print(f'  [{label}] uid={uid} ERROR: {e}')

    return np.array(feat_divs), np.array(logdets), np.array(n_hooks_list), np.array(accuracies)


print('\nScoring overrated architectures...')
over_fd, over_ld, over_nh, over_acc = score_group(overrated_uids, 'overrated')

print('\nScoring underrated architectures...')
under_fd, under_ld, under_nh, under_acc = score_group(underrated_uids, 'underrated')

# Results
print('\n' + '='*70)
print('RESULTS')
print('='*70)

print(f'\n{"Metric":<30} {"Overrated":>12} {"Underrated":>12} {"Diff":>10}')
print('-'*70)
print(f'{"Mean accuracy (%)":<30} {over_acc.mean():>12.2f} {under_acc.mean():>12.2f} {under_acc.mean()-over_acc.mean():>+10.2f}')
print(f'{"Mean logdet":<30} {over_ld[np.isfinite(over_ld)].mean():>12.2f} {under_ld[np.isfinite(under_ld)].mean():>12.2f} {under_ld[np.isfinite(under_ld)].mean()-over_ld[np.isfinite(over_ld)].mean():>+10.2f}')
print(f'{"Mean feature diversity":<30} {over_fd.mean():>12.2f} {under_fd.mean():>12.2f} {under_fd.mean()-over_fd.mean():>+10.2f}')
print(f'{"Mean n_hooks":<30} {over_nh.mean():>12.2f} {under_nh.mean():>12.2f} {under_nh.mean()-over_nh.mean():>+10.2f}')
print(f'{"Mean feat_div / n_hooks":<30} {(over_fd/np.maximum(over_nh,1)).mean():>12.2f} {(under_fd/np.maximum(under_nh,1)).mean():>12.2f} {(under_fd/np.maximum(under_nh,1)).mean()-(over_fd/np.maximum(over_nh,1)).mean():>+10.2f}')

# Plot distributions
SAVE_DIR = os.path.join('analysis_plots', 'failure_analysis')
os.makedirs(SAVE_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Feature diversity distributions
axes[0].hist(over_fd, bins=30, alpha=0.6, color='#D32F2F', label='Overrated', density=True)
axes[0].hist(under_fd, bins=30, alpha=0.6, color='#1976D2', label='Underrated', density=True)
axes[0].set_xlabel('Feature Diversity')
axes[0].set_ylabel('Density')
axes[0].set_title('Feature Diversity Distribution')
axes[0].legend()
axes[0].axvline(over_fd.mean(), color='#D32F2F', linestyle='--', alpha=0.8)
axes[0].axvline(under_fd.mean(), color='#1976D2', linestyle='--', alpha=0.8)

# Plot 2: Logdet distributions (for comparison)
over_ld_valid = over_ld[np.isfinite(over_ld)]
under_ld_valid = under_ld[np.isfinite(under_ld)]
axes[1].hist(over_ld_valid, bins=30, alpha=0.6, color='#D32F2F', label='Overrated', density=True)
axes[1].hist(under_ld_valid, bins=30, alpha=0.6, color='#1976D2', label='Underrated', density=True)
axes[1].set_xlabel('Log det K (NASWOT score)')
axes[1].set_ylabel('Density')
axes[1].set_title('NASWOT Score Distribution')
axes[1].legend()
axes[1].axvline(over_ld_valid.mean(), color='#D32F2F', linestyle='--', alpha=0.8)
axes[1].axvline(under_ld_valid.mean(), color='#1976D2', linestyle='--', alpha=0.8)

# Plot 3: Normalized feature diversity
over_fd_norm = over_fd / np.maximum(over_nh, 1)
under_fd_norm = under_fd / np.maximum(under_nh, 1)
axes[2].hist(over_fd_norm, bins=30, alpha=0.6, color='#D32F2F', label='Overrated', density=True)
axes[2].hist(under_fd_norm, bins=30, alpha=0.6, color='#1976D2', label='Underrated', density=True)
axes[2].set_xlabel('Feature Diversity / #ReLU hooks')
axes[2].set_ylabel('Density')
axes[2].set_title('Normalized Feature Diversity')
axes[2].legend()
axes[2].axvline(over_fd_norm.mean(), color='#D32F2F', linestyle='--', alpha=0.8)
axes[2].axvline(under_fd_norm.mean(), color='#1976D2', linestyle='--', alpha=0.8)

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Overrated vs Underrated: NASWOT fails because it ignores feature quality',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, '8_overrated_vs_underrated_feature_diversity.png'),
            dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(SAVE_DIR, '8_overrated_vs_underrated_feature_diversity.pdf'),
            bbox_inches='tight')
plt.close()
print(f'\nPlot saved to {SAVE_DIR}/8_overrated_vs_underrated_feature_diversity')
