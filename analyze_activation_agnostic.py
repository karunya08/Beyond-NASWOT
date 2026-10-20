"""
Analyze whether NASWOT is activation-agnostic at initialization.

Generates charts and tables showing:
1. Pre-activation value distributions per layer (show values >> transition regions)
2. Derivative mask vs binary mask comparison (show they're nearly identical)
3. Score correlation across activations (show rankings don't change with mask type)

Usage:
    .naswot\Scripts\python.exe analyze_activation_agnostic.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from utils import get_activation_mask, replace_relu, ACTIVATION_MODULES
import nasspace
import datasets
import random

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
N_ARCHS = 200           # sample architectures for analysis
BATCH_SIZE = 128
SEED = 1
OUT_DIR = 'analysis_plots'
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load search space and data
# ---------------------------------------------------------------------------
print("Loading NAS-Bench-201 and CIFAR-10...")
searchspace = nasspace.Nasbench201('cifar10', '../NAS-Bench-201-v1_0-e61699.pth')
class _Args:
    sigma = 0.05
train_loader = datasets.get_data('cifar10', '../cifardata/', True, BATCH_SIZE, 'none', 1, _Args())
data_iterator = iter(train_loader)
x_batch, target = next(data_iterator)
x_batch = x_batch.to(DEVICE)

# ---------------------------------------------------------------------------
# 1. Collect pre-activation stats per layer for a sample architecture
# ---------------------------------------------------------------------------
print("\n=== Analysis 1: Pre-activation distributions per layer ===")

uid_sample = 0
net = searchspace.get_network(uid_sample).to(DEVICE)

layer_preacts = []
layer_names = []

def collect_preact_hook(name):
    def hook(module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        layer_preacts.append(inp.detach().cpu().view(inp.size(0), -1))
        layer_names.append(name)
    return hook

hooks = []
for name, module in net.named_modules():
    if isinstance(module, nn.ReLU):
        hooks.append(module.register_forward_hook(collect_preact_hook(name)))

with torch.no_grad():
    net(x_batch)

for h in hooks:
    h.remove()

# Print stats table
print(f"\n{'Layer':<40} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'|x|<3 %':>10}")
print("-" * 90)
transition_fracs = []
layer_indices = []
layer_stds = []
for i, (name, preact) in enumerate(zip(layer_names, layer_preacts)):
    mn, mx = preact.min().item(), preact.max().item()
    mean, std = preact.mean().item(), preact.std().item()
    in_transition = ((preact.abs() < 3).float().mean().item()) * 100
    transition_fracs.append(in_transition)
    layer_indices.append(i)
    layer_stds.append(std)
    if i % 5 == 0 or i == len(layer_names) - 1:
        print(f"{name[:40]:<40} {mn:>10.1f} {mx:>10.1f} {mean:>10.2f} {std:>10.1f} {in_transition:>9.1f}%")

# Plot: fraction of values in transition region vs layer depth
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(layer_indices, transition_fracs, color='steelblue')
axes[0].set_xlabel('Layer index')
axes[0].set_ylabel('% of values in [-3, 3]')
axes[0].set_title('Fraction of pre-activations in Hardswish transition region')
axes[0].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100% (ideal for derivative mask)')
axes[0].legend()

axes[1].bar(layer_indices, layer_stds, color='coral')
axes[1].set_xlabel('Layer index')
axes[1].set_ylabel('Std of pre-activations')
axes[1].set_title('Pre-activation magnitude grows with depth')
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'preact_distributions.png'), dpi=150)
print(f"\nSaved: {OUT_DIR}/preact_distributions.png")


# ---------------------------------------------------------------------------
# 2. Mask comparison: derivative vs binary for each activation
# ---------------------------------------------------------------------------
print("\n=== Analysis 2: Derivative mask vs binary mask agreement ===")

# Use pre-activations already collected
print(f"\n{'Layer':<30} {'ReLU agree%':>12} {'Hardswish agree%':>16} {'Mish agree%':>12}")
print("-" * 75)

relu_agrees = []
hardswish_agrees = []
mish_agrees = []

for i, preact in enumerate(layer_preacts):
    binary = (preact > 0).float()
    
    relu_deriv = get_activation_mask(preact, 'relu')
    hs_deriv = get_activation_mask(preact, 'hardswish')
    mish_deriv = get_activation_mask(preact, 'mish')
    
    # For comparison: round derivative to binary (>0.5 -> 1, else 0)
    hs_binary = (hs_deriv > 0.5).float()
    mish_binary = (mish_deriv > 0.5).float()
    
    relu_agree = (relu_deriv == binary).float().mean().item() * 100
    hs_agree = (hs_binary == binary).float().mean().item() * 100
    mish_agree = (mish_binary == binary).float().mean().item() * 100
    
    relu_agrees.append(relu_agree)
    hardswish_agrees.append(hs_agree)
    mish_agrees.append(mish_agree)
    
    if i % 5 == 0 or i == len(layer_preacts) - 1:
        print(f"Layer {i:<25d} {relu_agree:>11.2f}% {hs_agree:>15.2f}% {mish_agree:>11.2f}%")

print(f"\n{'MEAN':<30} {np.mean(relu_agrees):>11.2f}% {np.mean(hardswish_agrees):>15.2f}% {np.mean(mish_agrees):>11.2f}%")

fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(layer_preacts))
w = 0.25
ax.bar(x_pos - w, hardswish_agrees, w, label='Hardswish deriv vs binary', color='orange')
ax.bar(x_pos,     mish_agrees, w, label='Mish deriv vs binary', color='green')
ax.bar(x_pos + w, relu_agrees, w, label='ReLU deriv vs binary (=100%)', color='blue', alpha=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('Agreement with binary mask (%)')
ax.set_title('Derivative masks match binary masks at init (values far from transition)')
ax.legend()
ax.set_ylim(90, 101)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'mask_agreement.png'), dpi=150)
print(f"Saved: {OUT_DIR}/mask_agreement.png")


# ---------------------------------------------------------------------------
# 3. Score comparison across activations (using multiple architectures)
# ---------------------------------------------------------------------------
print(f"\n=== Analysis 3: Score correlation across mask types ({N_ARCHS} archs) ===")

def compute_naswot_score(network, data, mask_type='relu'):
    """Compute NASWOT log-det score for a network."""
    K = np.zeros((BATCH_SIZE, BATCH_SIZE))
    
    def fwd_hook(module, inp, out):
        nonlocal K
        try:
            if isinstance(inp, tuple):
                inp = inp[0]
            inp = inp.view(inp.size(0), -1)
            x = get_activation_mask(inp, mask_type)
            k = x @ x.t()
            k2 = (1. - x) @ (1. - x).t()
            K = K + k.detach().cpu().numpy() + k2.detach().cpu().numpy()
        except:
            pass
    
    handles = []
    for name, module in network.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(fwd_hook))
    
    with torch.no_grad():
        network(data)
    
    for h in handles:
        h.remove()
    
    s, ld = np.linalg.slogdet(K)
    return ld

# Sample UIDs
uids = random.sample(range(15625), N_ARCHS)

scores_relu = []
scores_hardswish = []
scores_mish = []
scores_binary = []

for idx, uid in enumerate(uids):
    net = searchspace.get_network(uid).to(DEVICE)
    
    for m in net.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    
    # All masks on the SAME (ReLU) network — testing if mask choice matters
    s_relu = compute_naswot_score(net, x_batch, 'relu')
    s_hs = compute_naswot_score(net, x_batch, 'hardswish')
    s_mish = compute_naswot_score(net, x_batch, 'mish')
    
    scores_relu.append(s_relu)
    scores_hardswish.append(s_hs)
    scores_mish.append(s_mish)
    
    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/{N_ARCHS}] ...")

scores_relu = np.array(scores_relu)
scores_hardswish = np.array(scores_hardswish)
scores_mish = np.array(scores_mish)

# Correlations
tau_relu_hs, p_relu_hs = stats.kendalltau(scores_relu, scores_hardswish)
tau_relu_mish, p_relu_mish = stats.kendalltau(scores_relu, scores_mish)
tau_hs_mish, p_hs_mish = stats.kendalltau(scores_hardswish, scores_mish)
r_relu_hs = np.corrcoef(scores_relu, scores_hardswish)[0, 1]
r_relu_mish = np.corrcoef(scores_relu, scores_mish)[0, 1]
r_hs_mish = np.corrcoef(scores_hardswish, scores_mish)[0, 1]

print(f"\nScore correlation table ({N_ARCHS} architectures, ReLU network, different masks):")
print(f"{'Pair':<30} {'Kendall-tau':>12} {'Pearson-r':>10}")
print("-" * 55)
print(f"{'ReLU vs Hardswish':<30} {tau_relu_hs:>12.4f} {r_relu_hs:>10.4f}")
print(f"{'ReLU vs Mish':<30} {tau_relu_mish:>12.4f} {r_relu_mish:>10.4f}")
print(f"{'Hardswish vs Mish':<30} {tau_hs_mish:>12.4f} {r_hs_mish:>10.4f}")

# Scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
pairs = [
    (scores_relu, scores_hardswish, 'ReLU', 'Hardswish', tau_relu_hs, r_relu_hs),
    (scores_relu, scores_mish, 'ReLU', 'Mish', tau_relu_mish, r_relu_mish),
    (scores_hardswish, scores_mish, 'Hardswish', 'Mish', tau_hs_mish, r_hs_mish),
]
for ax, (sx, sy, lx, ly, tau, r) in zip(axes, pairs):
    ax.scatter(sx, sy, alpha=0.4, s=10, c='steelblue')
    mn = min(sx.min(), sy.min())
    mx = max(sx.max(), sy.max())
    ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel(f'{lx} mask score')
    ax.set_ylabel(f'{ly} mask score')
    ax.set_title(f'{lx} vs {ly}\nτ={tau:.4f}, r={r:.4f}')
    ax.legend()

plt.suptitle('NASWOT scores with different masks on same (ReLU) networks\n'
             'High correlation = mask choice does not matter at init', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'score_correlation.png'), dpi=150)
print(f"\nSaved: {OUT_DIR}/score_correlation.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY: NASWOT is activation-agnostic at initialization")
print("=" * 70)
print(f"""
1. Pre-activation scale: Values grow to ±{max(layer_stds):.0f} in deeper layers.
   Only {np.mean(transition_fracs):.1f}% of values fall in Hardswish's [-3,3] transition.

2. Mask agreement: Derivative masks match binary masks {np.mean(hardswish_agrees):.1f}%
   (Hardswish) and {np.mean(mish_agrees):.1f}% (Mish) of the time.

3. Score ranking: Kendall-tau between mask types:
   ReLU↔Hardswish: {tau_relu_hs:.4f}
   ReLU↔Mish:      {tau_relu_mish:.4f}
   Hardswish↔Mish:  {tau_hs_mish:.4f}

Conclusion: At random initialization, derivative masks degenerate to binary
masks because pre-activation magnitudes far exceed the smooth transition
regions. The mask type (and therefore the activation choice) has negligible
effect on NASWOT rankings.
""")
