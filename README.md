# Beyond NASWOT

**Course**: C6886 | **Team**: Karunya Harikrishnan, Tejaswi  
Based on [Neural Architecture Search Without Training](https://github.com/BayesWatch/nas-without-training) (Mellor et al., ICML 2021)

---

## Problem

NASWOT scores untrained architectures using activation-pattern diversity (log det of kernel matrix K). Two problems:

1. **Activation-specific**: Binary masks (`1[h>0]`) are ReLU-specific
2. **Skip-blind**: NASWOT systematically underrates architectures with skip connections (rank error of -2100 positions for true-residual architectures)

## Our Contributions

### Finding 1: NASWOT is Activation-Agnostic at Random Init (P1/P2)

We implemented derivative masks (`sigma'(h)`) for HardSwish and Mish (alongside ReLU). Key insight: in untrained networks, pre-activation magnitudes grow with depth (~4000 by layer 15). The smooth transition region of Swish/Mish lies in [-3, 3] -- essentially no values land there. All asymptotically-linear activations (ReLU, Swish, Mish, GELU, ELU) degenerate to the same binary step function at random init.

**Implication**: NASWOT already works for all mainstream activations without modification. The exception is doubly-saturating activations (tanh, sigmoid), for which it degenerates entirely.

### Finding 2: Feature Diversity Beats NASWOT (tau +0.10)

| Metric            | CIFAR-10    | CIFAR-100   |
| ----------------- | ----------- | ----------- |
| NASWOT (logdet)   | 0.5634      | 0.5921      |
| Feature Diversity | 0.6704      | 0.6976      |
| **Improvement**   | **+0.1070** | **+0.1055** |

Feature diversity = sum of effective ranks of channel correlation matrices across Conv2d layers. The improvement is largest in the top half where NAS actually needs to discriminate (top-half tau: +0.068 on CIFAR-10, +0.126 on CIFAR-100).

### Finding 3: NASWOT's Failure Mode is Skip Blindness

| Feature           | tau(feat, acc) | tau(feat, score) | Gap                        |
| ----------------- | -------------- | ---------------- | -------------------------- |
| n_parametric      | +0.55          | +0.81            | NASWOT loves params        |
| n_skip            | -0.05          | -0.24            | NASWOT penalizes 5x harder |
| has_true_residual | +0.03          | -0.15            | NASWOT penalizes residuals |

Overrated architectures (high score, low accuracy) have 0.6% true residual. Underrated (low score, high accuracy) have 47%. NASWOT's top 50 picks are **100% zero-skip**.

### Finding 4: Gradient Consistency Detects Skip Connections

Gradient Consistency (GC) = cosine similarity of gradients on two mini-batches. Within NASWOT's top quartile:

| Split            | Mean Acc | Mean Skips | % True Residual |
| ---------------- | -------- | ---------- | --------------- |
| High GC (n=1928) | 89.02%   | 0.82       | **18.1%**       |
| Low GC (n=1928)  | 88.92%   | 0.69       | **0.7%**        |
| Ground Truth     | 89.66%   | 1.22       | 27.0%           |

- GC vs skip count: tau=0.11 (NASWOT pool), tau=0.21 (FeatDiv pool), both p<0.0001
- Accuracy diff: p=0.012 (significant on NASWOT pool)
- True residual: 26x difference between high/low GC halves

### Finding 5: Two-Stage Pipeline

**FeatDiv + GC** nearly perfectly recovers the ground truth skip distribution:

| Method                | Mean Skips | % True Residual |
| --------------------- | ---------- | --------------- |
| NASWOT top-Q          | 0.75       | 9.4%            |
| FeatDiv top-Q         | 0.89       | 13.7%           |
| FeatDiv + GC top half | 1.04       | **26.5%**       |
| Ground Truth top-Q    | 1.22       | **26.8%**       |

---

## Setup

### Option 1: Conda (recommended)

```bash
conda config --set channel_priority strict
conda env create -f env.yml --solver=libmamba
conda activate naswot
```

`env.yml` uses the `pytorch`, `nvidia` and `defaults` channels only. Do **not** add `conda-forge` (causes DLL load failures on Windows).

### Option 2: venv + pip

```bash
python -m venv .naswot
.naswot\Scripts\activate      # Windows
source .naswot/bin/activate   # Linux/Mac
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy>=1.24,<2" scipy matplotlib tqdm yacs thop
```

### Data & Benchmark

1. **CIFAR-10/100**: Downloaded automatically on first run to `../cifardata/`
2. **NAS-Bench-201 API** (~2 GB): Download `NAS-Bench-201-v1_0-e61699.pth` from the [NAS-Bench-201 repository](https://github.com/D-X-Y/NAS-Bench-201) and place at `../NAS-Bench-201-v1_0-e61699.pth` (or pass `--api_loc <path>`)

---

## Usage

### Activation-agnostic scoring (P1/P2)

```bash
# Baseline: binary masks, ReLU (original NASWOT)
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet \
    --batch_size 128 --GPU 0 --dataset cifar10 --method binary_tau

# Derivative masks, ReLU (should match baseline exactly)
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet \
    --batch_size 128 --GPU 0 --dataset cifar10 --method masks --mask_type relu

# Derivative masks, HardSwish
python score_networks.py ... --method masks --mask_type hardswish

# Derivative masks, Mish
python score_networks.py ... --method masks --mask_type mish

# Different datasets: cifar10 | cifar100 | ImageNet16-120
python score_networks.py ... --dataset cifar100

# Quick test (first 50 architectures only)
python score_networks.py ... --max_archs 50
```

### Feature diversity & pipeline

```bash
# Feature diversity scoring (all 15,625 architectures, ~30 min)
python score_feature_diversity_full.py --trainval --GPU 0

# Two-stage pipeline: primary metric top quartile + GC re-rank (~10 min)
python search_pipeline.py --trainval --GPU 0                        # NASWOT + GC
python search_pipeline.py --trainval --GPU 0 --primary feat_div     # FeatDiv + GC

# Generate plots from pipeline results
python plots/plot_pipeline.py --results results/pipeline_naswot_topk3856_gc.npy.npz
python plots/plot_pipeline.py --results results/pipeline_feat_div_topk3906_gc.npy.npz
```

### Failure analysis

```bash
# Identify overrated/underrated architectures (requires pre-computed NASWOT scores)
python experiments/failure_analysis.py \
    --scores results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy \
    --accs results/full_baseline_accs_nasbench201_cifar10_True.npy

# Structural attribution (requires failure_sets/ from above)
python experiments/failure_attribution.py

# Feature diversity comparison of overrated vs underrated
python experiments/compare_overrated_underrated.py --trainval --GPU 0

# Stratified analysis by skip count (requires pre-computed NASWOT scores)
python experiments/skip_stratified_analysis.py

# Activation agnosticity analysis
python experiments/analyze_activation_agnostic.py --GPU 0
```

### Plots

```bash
# Pipeline results (requires search_pipeline.py output)
python plots/plot_pipeline.py --results results/pipeline_topk3856_gc.npy.npz

# Gradient consistency (requires test_grad_consistency.py output)
python plots/plot_grad_consistency.py

# Failure analysis (requires pre-computed NASWOT scores)
python plots/plot_failure_analysis.py \
    --scores results/full_baseline_hook_logdet_nasbench201_cifar10__none_0.05_1_True_128_1_1.npy \
    --accs results/full_baseline_accs_nasbench201_cifar10_True.npy
```

---

## Project Structure

```
Beyond-NASWOT/
├── score_networks.py               # NASWOT scoring + P1/P2 (derivative masks, binary threshold)
├── score_feature_diversity_full.py  # Feature diversity for all 15,625 architectures
├── search_pipeline.py              # Two-stage pipeline: primary metric + GC re-rank
├── score_networks_expressivity.py  # P3: layer-weighted spectral scoring
├── scores.py                       # Score functions (logdet, eff_rank, condition, spectral_gap)
├── utils.py                        # Activation masks, ReLU replacement utilities
├── nasspace.py                     # NAS-Bench-201 search space wrapper
├── env.yml                         # Conda environment spec
│
├── plots/                          # Visualization scripts
│   ├── plot_pipeline.py            # Pipeline result plots (GC split, accuracy comparison)
│   ├── plot_grad_consistency.py    # Gradient consistency plots
│   ├── plot_failure_analysis.py    # Failure analysis plots
│   ├── plot_skip_metrics.py        # Skip connection metric plots
│   └── plot_scores.py             # Original score vs accuracy scatter
│
├── experiments/                    # Analysis & experiment scripts
│   ├── test_feature_diversity.py   # 200-arch feature diversity prototype
│   ├── test_grad_consistency.py    # 200-arch gradient consistency experiment
│   ├── compare_overrated_underrated.py  # Overrated vs underrated topology analysis
│   ├── skip_stratified_analysis.py      # Full stratified analysis by skip count
│   ├── analyze_activation_agnostic.py   # Activation agnosticity proof
│   ├── failure_analysis.py         # NASWOT failure case identification
│   └── failure_attribution.py      # Structural attribution of failures
│
├── results/                        # Pre-computed scores (.npy) and pipeline outputs (.npz)
├── models/                         # NAS-Bench-201 cell-based tiny networks
├── datasets/                       # Dataset loading (CIFAR-10/100, ImageNet16-120)
└── nas_201_api/                    # NAS-Bench-201 Python API
```

## References

- Mellor et al., "Neural Architecture Search without Training", ICML 2021
- Abdelfattah et al., "Zero-Cost Proxies for Lightweight NAS", ICLR 2021
- Dong & Yang, "NAS-Bench-201", ICLR 2020
- Kanadiya et al., "ReLU is all you need for NASWOT", AutoML 2025
