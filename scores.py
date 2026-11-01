import numpy as np
import torch


# ---------------------------------------------------------------------------
# Original scores
# ---------------------------------------------------------------------------

def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


# ---------------------------------------------------------------------------
# Spectral scores  (P3b -- eigenvalue decomposition of K)
# ---------------------------------------------------------------------------

def _safe_eigvals(K):
    """Return eigenvalues of K sorted ascending, clamped for stability."""
    eigvals = np.linalg.eigvalsh(K)
    return np.maximum(eigvals, 1e-10)


def hook_effective_rank(K, labels=None):
    """Effective rank = exp(entropy of normalised eigenvalue distribution).

    Captures how many independent directions the activation patterns span.
    Higher effective rank -> more uniform use of representation space.
    """
    eigvals = _safe_eigvals(K)
    p = eigvals / eigvals.sum()
    entropy = -np.sum(p * np.log(p))
    return entropy  # log of effective rank; monotonic, avoids large numbers


def hook_condition(K, labels=None):
    """Negative log condition number of K.

    Condition number = lambda_max / lambda_min.  Lower condition -> better
    conditioned kernel -> more stable training.  We negate so that *higher
    score = better architecture* (consistent with other scores).
    """
    eigvals = _safe_eigvals(K)
    cond = eigvals[-1] / eigvals[0]
    return -np.log(cond)


def hook_spectral_gap(K, labels=None):
    """Ratio of second-largest to largest eigenvalue (closer to 1 = more uniform).

    Large spectral gap means one direction dominates; small gap means the
    kernel distributes information across multiple dimensions.
    """
    eigvals = _safe_eigvals(K)
    if len(eigvals) < 2:
        return 0.0
    return eigvals[-2] / eigvals[-1]


# ---------------------------------------------------------------------------
# CKA helpers  (for layer-weighted scoring)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CKA-guided layer weighting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Score registries
# ---------------------------------------------------------------------------

# K-based scores (operate on full K matrix)
K_SCORES = {
    'hook_logdet': hooklogdet,
}

# Layer-based scores (operate on per-layer K matrices)
LAYER_SCORES = {
    'layer_cka_weighted': layer_cka_weighted,
}

# Legacy registry (used by score_networks.py)
_scores = {
        'hook_logdet': hooklogdet,
        'hook_eff_rank': hook_effective_rank,
        'hook_condition': hook_condition,
        'hook_spectral_gap': hook_spectral_gap,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]
