"""
score_cka_systems.py
====================
Score all NAS-Bench-201 architectures with CKA-weighted + systems-aware methods.

For each architecture in a single pass:
  1. Collect per-layer K matrices (ReLU hooks)
  2. Compute baseline score: log det(sum of K_layers)
  3. Compute CKA-weighted score: label-guided layer weighting
  4. Measure hardware: params, latency, peak GPU memory
  5. Compute systems scores: expressivity / cost^alpha

Usage:
    # CIFAR-10 full run (~60 min)
    python score_cka_systems.py --trainval --GPU 0 --dataset cifar10 \
        --api_loc <path-to-NAS-Bench-201.pth> --data_loc ./cifardata

    # CIFAR-100
    python score_cka_systems.py --trainval --GPU 0 --dataset cifar100 ...

    # Quick test (500 architectures, ~5 min)
    python score_cka_systems.py --trainval --GPU 0 --max_archs 500 ...

    # Custom alpha for systems scoring
    python score_cka_systems.py --trainval --GPU 0 --alpha 0.3 ...

Results are saved to results/ as .npy files. Use analyze_cka_systems.py
to load results, print tables, and generate plots.
"""

import argparse
import nasspace
import datasets
import random
import numpy as np
import os
import time
import torch
from scores import (K_SCORES, LAYER_SCORES,
                    systems_score_params, systems_score_latency,
                    systems_score_memory)
from scipy import stats
from utils import add_dropout, init_network

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Beyond NASWOT — Novel Scoring')
parser.add_argument('--data_loc', default='../cifardata/', type=str)
parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str)
parser.add_argument('--save_loc', default='results', type=str)
parser.add_argument('--save_string', default='v2', type=str)
parser.add_argument('--nasspace', default='nasbench201', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--augtype', default='none', type=str)
parser.add_argument('--sigma', default=0.05, type=float)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--init', default='', type=str)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dropout', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--maxofn', default=1, type=int)
parser.add_argument('--stem_out_channels', default=16, type=int)
parser.add_argument('--num_stacks', default=3, type=int)
parser.add_argument('--num_modules_per_stack', default=3, type=int)
parser.add_argument('--num_labels', default=1, type=int)
parser.add_argument('--max_archs', default=-1, type=int,
                    help='Max architectures to score (-1 = all). Use 500 for quick test.')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='Systems scoring cost exponent (0=ignore cost, 1=strong cost penalty)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y, out = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach(), out.detach()


# ── Setup ────────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f'Using GPU {args.GPU}: {torch.cuda.get_device_name(0)} '
          f'(PyTorch CUDA {torch.version.cuda})', flush=True)
else:
    print(f'Using CPU (PyTorch {torch.__version__})', flush=True)

searchspace = nasspace.get_search_space(args)
savedataset = args.dataset
args.dataset = args.dataset.replace('fake', '')
if args.dataset == 'cifar10':
    args.dataset = args.dataset + '-valid'
if 'valid' in args.dataset:
    args.dataset = args.dataset.replace('-valid', '')
train_loader = datasets.get_data(
    args.dataset, args.data_loc, args.trainval,
    args.batch_size, args.augtype, args.repeat, args
)
os.makedirs(args.save_loc, exist_ok=True)

if args.dataset == 'cifar10':
    acc_type = 'ori-test'
else:
    acc_type = 'x-test'

# ── Methods to evaluate ─────────────────────────────────────────────────────
ALL_K = list(K_SCORES.keys())
ALL_LAYER = list(LAYER_SCORES.keys())
ALL_METHODS = ALL_K + ALL_LAYER

# Systems-aware method names (computed post-hoc from CKA score + hardware metrics)
alpha = args.alpha
SYSTEMS_METHODS = [
    f'sys_params_{alpha}',
    f'sys_latency_{alpha}',
    f'sys_memory_{alpha}',
]

N_total = len(searchspace)
N = min(args.max_archs, N_total) if args.max_archs > 0 else N_total
all_scores = {m: np.zeros(N) for m in ALL_METHODS + SYSTEMS_METHODS}
hw_params = np.zeros(N)
hw_latency = np.zeros(N)
hw_memory = np.zeros(N)

accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}'
try:
    accs = np.load(accfilename + '.npy')
    if len(accs) != N:
        accs = np.zeros(N)
except Exception:
    accs = np.zeros(N)

print_interval = 250 if N <= 3000 else 500
print(f'\nScoring {N}/{N_total} architectures')
print(f'  Expressivity methods: {ALL_K + ALL_LAYER}')
print(f'  Systems methods:      {SYSTEMS_METHODS}')
print(f'  Total: {len(ALL_METHODS) + len(SYSTEMS_METHODS)} methods\n', flush=True)

# ── Main loop ────────────────────────────────────────────────────────────────
for i, (uid, network) in enumerate(searchspace):
    if i >= N:
        break
    try:
        if args.dropout:
            add_dropout(network, args.sigma)
        if args.init != '':
            init_network(network, args.init)

        network.K = np.zeros((args.batch_size, args.batch_size))
        network.K_layers = []

        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K_l = (x @ x.t() + (1. - x) @ (1. - x.t())).cpu().numpy()
                network.K_layers.append(K_l)
                network.K = network.K + K_l
            except Exception:
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

        network = network.to(device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # ── Measure hardware metrics ──
        hw_params[i] = sum(p.numel() for p in network.parameters())

        data_iterator = iter(train_loader)
        x, target = next(data_iterator)
        x2 = torch.clone(x)
        x2 = x2.to(device)
        x, target = x.to(device), target.to(device)

        # Backward pass (marks visited modules)
        jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)

        # Forward pass with latency + memory measurement
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        network(x2.to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        hw_latency[i] = time.perf_counter() - t_start
        if device.type == 'cuda':
            hw_memory[i] = torch.cuda.max_memory_allocated()

        # ── Expressivity scores ──
        for method in ALL_K:
            all_scores[method][i] = K_SCORES[method](network.K, target)
        if network.K_layers:
            for method in ALL_LAYER:
                all_scores[method][i] = LAYER_SCORES[method](network.K_layers, target)

        # ── Systems-aware scores (using CKA expressivity + hardware cost) ──
        expr = all_scores['layer_cka_weighted'][i]
        all_scores[f'sys_params_{alpha}'][i] = systems_score_params(expr, hw_params[i], alpha=alpha)
        all_scores[f'sys_latency_{alpha}'][i] = systems_score_latency(expr, hw_latency[i], alpha=alpha)
        all_scores[f'sys_memory_{alpha}'][i] = systems_score_memory(expr, hw_memory[i], alpha=alpha)

        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)

        # Print progress
        if (i + 1) % print_interval == 0 or i == 0:
            s_base = all_scores['hook_logdet'][:i+1]
            a = accs[:i+1]
            valid_base = ~np.isnan(s_base) & (s_base != 0)
            tau_base = stats.kendalltau(a[valid_base], s_base[valid_base])[0] if valid_base.sum() > 1 else float('nan')

            print(f'\n  Arch {i+1}/{N}  |  baseline tau = {tau_base:+.4f}', flush=True)
            for method in ALL_LAYER + SYSTEMS_METHODS:
                s = all_scores[method][:i+1]
                valid = ~np.isnan(s) & (s != 0)
                tau = stats.kendalltau(a[valid], s[valid])[0] if valid.sum() > 1 else float('nan')
                diff = tau - tau_base if not (np.isnan(tau) or np.isnan(tau_base)) else float('nan')
                marker = ' ***' if diff > 0.005 else ''
                print(f'    {method:>25s}  tau={tau:+.4f}  ({diff:+.4f}){marker}', flush=True)

            # Hardware stats summary
            p = hw_params[:i+1]
            print(f'    --- HW: params {p.mean()/1e3:.0f}K [{p.min()/1e3:.0f}K-{p.max()/1e3:.0f}K]'
                  f'  latency {hw_latency[:i+1].mean()*1e3:.1f}ms'
                  f'  memory {hw_memory[:i+1].mean()/1e6:.0f}MB', flush=True)

        if i % 2000 == 0:
            for method in ALL_METHODS + SYSTEMS_METHODS:
                np.save(f'{args.save_loc}/{args.save_string}_{method}_{args.nasspace}_{savedataset}_{args.trainval}_{args.seed}', all_scores[method])
            np.save(f'{args.save_loc}/{args.save_string}_hw_params_{args.nasspace}_{savedataset}_{args.trainval}', hw_params)
            np.save(f'{args.save_loc}/{args.save_string}_hw_latency_{args.nasspace}_{savedataset}_{args.trainval}', hw_latency)
            np.save(f'{args.save_loc}/{args.save_string}_hw_memory_{args.nasspace}_{savedataset}_{args.trainval}', hw_memory)
            np.save(accfilename, accs)

        del network.K_layers

    except Exception as e:
        print(f'arch {i}: {e}')
        accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        for method in ALL_METHODS + SYSTEMS_METHODS:
            all_scores[method][i] = np.nan

# ── Save all results ─────────────────────────────────────────────────────────
np.save(accfilename, accs)
np.save(f'{args.save_loc}/{args.save_string}_hw_params_{args.nasspace}_{savedataset}_{args.trainval}', hw_params)
np.save(f'{args.save_loc}/{args.save_string}_hw_latency_{args.nasspace}_{savedataset}_{args.trainval}', hw_latency)
np.save(f'{args.save_loc}/{args.save_string}_hw_memory_{args.nasspace}_{savedataset}_{args.trainval}', hw_memory)

# ── Final expressivity results ───────────────────────────────────────────────
s_base = all_scores['hook_logdet']
valid_base = ~np.isnan(s_base) & (s_base != 0)
tau_base = stats.kendalltau(accs[valid_base], s_base[valid_base])[0] if valid_base.sum() > 1 else float('nan')

all_method_names = ALL_METHODS + SYSTEMS_METHODS
results = []
for method in all_method_names:
    s = all_scores[method]
    valid = ~np.isnan(s) & (s != 0)
    tau, p = stats.kendalltau(accs[valid], s[valid]) if valid.sum() > 1 else (float('nan'), float('nan'))
    diff = tau - tau_base if not (np.isnan(tau) or np.isnan(tau_base)) else float('nan')
    results.append((method, tau, p, diff))
    np.save(f'{args.save_loc}/{args.save_string}_{method}_{args.nasspace}_{savedataset}_{args.trainval}_{args.seed}', all_scores[method])

results.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)

print(f'\n{"="*70}')
print(f'  FINAL RESULTS  ({N} architectures, {savedataset})')
print(f'  Baseline tau = {tau_base:+.4f}')
print(f'{"="*70}')
print(f'  {"Rank":>4s}  {"Method":>25s}  {"tau":>8s}  {"delta":>8s}')
print(f'  {"-"*50}')
for rank, (method, tau, p, diff) in enumerate(results, 1):
    marker = ' <-- BEST' if rank == 1 else ''
    print(f'  {rank:>4d}  {method:>25s}  {tau:>+8.4f}  {diff:>+8.4f}{marker}')
print(f'{"="*70}')

# ── Systems analysis: Top architecture picks ─────────────────────────────────
print(f'\n{"="*70}')
print(f'  SYSTEMS ANALYSIS — Which architecture does each method pick?')
print(f'{"="*70}')
print(f'  {"Method":>25s}  {"Accuracy":>8s}  {"Params":>8s}  {"Latency":>10s}  {"Memory":>10s}')
print(f'  {"-"*65}')

key_methods = ['hook_logdet', 'layer_cka_weighted'] + SYSTEMS_METHODS
for method in key_methods:
    s = all_scores[method]
    valid = ~np.isnan(s) & (s != 0)
    if valid.sum() == 0:
        continue
    # Find top-1 architecture by this method's score
    best_idx = np.nanargmax(np.where(valid, s, -np.inf))
    acc = accs[best_idx]
    par = hw_params[best_idx]
    lat = hw_latency[best_idx]
    mem = hw_memory[best_idx]
    print(f'  {method:>25s}  {acc:>7.2f}%  {par/1e3:>7.0f}K  {lat*1e3:>8.1f}ms  {mem/1e6:>8.0f}MB')

# Best accuracy architecture (oracle)
best_acc_idx = np.argmax(accs)
print(f'  {"(oracle best acc)":>25s}  {accs[best_acc_idx]:>7.2f}%  {hw_params[best_acc_idx]/1e3:>7.0f}K  '
      f'{hw_latency[best_acc_idx]*1e3:>8.1f}ms  {hw_memory[best_acc_idx]/1e6:>8.0f}MB')
print(f'{"="*70}')
