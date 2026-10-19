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
# Score registry
# ---------------------------------------------------------------------------

_scores = {
        'hook_logdet': hooklogdet,
        'hook_eff_rank': hook_effective_rank,
        'hook_condition': hook_condition,
        'hook_spectral_gap': hook_spectral_gap,
        'random': random_score
        }

def get_score_func(score_name):
    return _scores[score_name]
