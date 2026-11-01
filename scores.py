"""
Scoring functions for NASWOT and extensions.

Baseline:  hooklogdet  (Mellor et al., ICML 2021)
Novelty:   CKA-guided layer weighting — use class labels to determine
           which layers matter via Centered Kernel Alignment
Systems:   Hardware-cost-aware scoring — normalize expressivity by
           real hardware metrics (params, latency, memory)
"""

import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_numpy(labels):
    """Convert labels tensor to numpy array."""
    if hasattr(labels, 'cpu'):
        labels = labels.cpu()
    if hasattr(labels, 'numpy'):
        labels = labels.numpy()
    return labels.ravel()


def _weighted_K(K_layers, weights):
    """Combine per-layer K matrices with given weights."""
    K = np.zeros_like(K_layers[0])
    for w, K_l in zip(weights, K_layers):
        K += w * K_l
    return K


def _cka(K, Y):
    """Centered Kernel Alignment between two kernel matrices."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Yc = H @ Y @ H
    denom = np.linalg.norm(Kc, 'fro') * np.linalg.norm(Yc, 'fro')
    if denom < 1e-10:
        return 0.0
    return np.trace(Kc @ Yc) / denom


# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINE — Original NASWOT (Mellor et al., ICML 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def hooklogdet(K, labels=None):
    """Original NASWOT: log det(K).  Label-agnostic, equal layer weights."""
    s, ld = np.linalg.slogdet(K)
    return ld


# ═══════════════════════════════════════════════════════════════════════════════
#  CKA-GUIDED LAYER WEIGHTING
#
#  Use CKA(K_l, Y) to let labels discover which layers matter.
#  Layers whose activation patterns correlate with class structure
#  get higher weight.
# ═══════════════════════════════════════════════════════════════════════════════

def layer_cka_weighted(K_layers, labels):
    """CKA-guided layer weighting.

    For each layer, compute CKA(K_l, Y) where Y is the ideal label kernel.
    Use CKA as the layer weight.  Layers already correlated with class
    structure contribute more to the final score.
    """
    labels = _to_numpy(labels)
    Y = (labels[:, None] == labels[None, :]).astype(np.float64)
    L = len(K_layers)
    if L == 0:
        return 0.0
    weights = [max(_cka(K_l, Y), 0.0) for K_l in K_layers]
    if sum(weights) < 1e-10:
        weights = [1.0] * L
    K = _weighted_K(K_layers, weights)
    _, ld = np.linalg.slogdet(K)
    return ld


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEMS COMPONENT — HARDWARE-COST-AWARE SCORING
#
#  Incorporate real hardware metrics (params, latency, memory) into the score.
#  Instead of "which architecture is most expressive?", ask "which gives the
#  best expressivity per unit of hardware cost?"
#
#  systems_score = expressivity / cost^alpha
#
#  These are post-processing functions applied AFTER computing expressivity.
# ═══════════════════════════════════════════════════════════════════════════════

def systems_score_params(expressivity, params, alpha=0.5):
    """Expressivity normalized by parameter count.

    score = expressivity / params^alpha

    Favors architectures that are expressive with fewer parameters.
    """
    if params <= 0:
        return expressivity
    return expressivity / (params ** alpha)


def systems_score_latency(expressivity, latency, alpha=0.5):
    """Expressivity normalized by inference latency.

    score = expressivity / latency^alpha

    Favors architectures that are expressive AND fast on the target GPU.
    This is the most "systems-aware" metric because latency captures real
    hardware behavior (memory bandwidth, kernel efficiency, parallelism)
    that FLOPs cannot.
    """
    if latency <= 0:
        return expressivity
    return expressivity / (latency ** alpha)


def systems_score_memory(expressivity, memory, alpha=0.5):
    """Expressivity normalized by peak GPU memory.

    score = expressivity / memory^alpha

    Favors architectures that achieve expressivity within a memory budget.
    """
    if memory <= 0:
        return expressivity
    return expressivity / (memory ** alpha)



# ─── Registries ──────────────────────────────────────────────────────────────

# K-based scores (operate on full K matrix; signature: K, labels)
K_SCORES = {
    'hook_logdet': hooklogdet,
}

# Layer-based scores (operate on per-layer K matrices; signature: K_layers, labels)
LAYER_SCORES = {
    'layer_cka_weighted': layer_cka_weighted,
}

# Legacy registry for backward compatibility with baseline score_networks.py
_scores = {
    'hook_logdet': hooklogdet,
}

def get_score_func(score_name):
    return _scores[score_name]
