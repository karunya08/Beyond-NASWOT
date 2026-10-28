"""
score_feature_diversity_full.py
===============================
Score ALL 15,625 NAS-Bench-201 architectures with feature diversity
(sum of effective ranks across conv layers). Saves results for use
by search_pipeline.py.

Usage:
    python score_feature_diversity_full.py --trainval --GPU 0
"""

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import torch
import time
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
train_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval,
                                  args.batch_size, 'none', 1, args)

acc_type = 'ori-test' if args.dataset == 'cifar10' else 'x-test'


def effective_rank(M):
    """Effective rank via SVD entropy."""
    s = np.linalg.svd(M, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return np.exp(-np.sum(p * np.log(p + 1e-15)))


# Load one data batch (same for all architectures)
data_iterator = iter(train_loader)
x_batch, _ = next(data_iterator)
x_batch = x_batch.to(device)

n_archs = len(searchspace)
scores = np.zeros(n_archs)
accs_arr = np.zeros(n_archs)

# Try to load existing accuracies
acc_file = f'{args.save_loc}/full_baseline_accs_nasbench201_cifar10_True.npy'
if os.path.exists(acc_file):
    accs_arr = np.load(acc_file)
    print(f'Loaded pre-computed accuracies from {acc_file}')

print(f'Scoring {n_archs} architectures with feature diversity...\n')
start_time = time.time()

for i, (uid, network) in enumerate(searchspace):
    try:
        fmaps = []

        def hook_fn(m, inp, out):
            if out.dim() == 4 and out.shape[1] > 1:
                fmaps.append(out.detach())

        for m in network.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        hooks = [m.register_forward_hook(hook_fn)
                 for n, m in network.named_modules()
                 if isinstance(m, torch.nn.Conv2d)]

        network = network.to(device)
        network.eval()
        with torch.no_grad():
            network(x_batch)

        total = 0.0
        for fm in fmaps:
            flat = fm.mean(0).reshape(fm.shape[1], -1).cpu().numpy()
            if flat.shape[0] > 1 and flat.shape[1] > 1:
                total += effective_rank(flat)

        scores[i] = total

        for h in hooks:
            h.remove()

        if accs_arr[i] == 0:
            accs_arr[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)

        del network
        torch.cuda.empty_cache()

    except Exception as e:
        scores[i] = 0
        if accs_arr[i] == 0:
            accs_arr[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)

    if (i + 1) % 500 == 0 or i == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        remaining = (n_archs - i - 1) / rate
        valid = (scores[:i+1] > 0) & (accs_arr[:i+1] > 0)
        if valid.sum() > 2:
            tau, _ = stats.kendalltau(accs_arr[:i+1][valid], scores[:i+1][valid])
        else:
            tau = float('nan')
        print(f'[{i+1:>6}/{n_archs}] tau={tau:.4f} | '
              f'{rate:.1f} arch/s | ETA {remaining/60:.1f} min', flush=True)

    # Checkpoint every 5000
    if (i + 1) % 5000 == 0:
        np.save(f'{args.save_loc}/full_feat_div_scores.npy', scores)
        np.save(f'{args.save_loc}/full_feat_div_accs.npy', accs_arr)

# Final save
os.makedirs(args.save_loc, exist_ok=True)
np.save(f'{args.save_loc}/full_feat_div_scores.npy', scores)
np.save(f'{args.save_loc}/full_feat_div_accs.npy', accs_arr)

# Final stats
valid = (scores > 0) & (accs_arr > 0)
tau, p = stats.kendalltau(accs_arr[valid], scores[valid])
elapsed = time.time() - start_time

print(f'\n{"="*60}')
print(f'DONE in {elapsed/60:.1f} minutes')
print(f'Valid: {valid.sum()}/{n_archs}')
print(f'Kendall tau (feat_div, acc): {tau:.4f} (p={p:.2e})')
print(f'Saved to {args.save_loc}/full_feat_div_scores.npy')
print(f'{"="*60}')
