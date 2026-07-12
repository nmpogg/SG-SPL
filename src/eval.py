"""
Evaluation metrics for ZS-SBIR and GZS-SBIR.

Metrics:
    mAP@all  — mean Average Precision over all gallery items
    mAP@K    — mean Average Precision truncated at K (e.g. K=200)
    P@K      — Precision at K (e.g. K=100, K=200)

Standard protocol:
    ZS-SBIR:  query=sketch (unseen), gallery=photo (unseen only)
    GZS-SBIR: query=sketch (unseen), gallery=photo (seen + unseen)

Formula for AP@K (standard IR definition):
    AP@K = sum_{i=1}^{K} [P(i) * rel(i)] / min(total_relevant, K)

    where P(i) is Precision at rank i, rel(i)=1 if item at rank i is relevant.
    The denominator min(total_relevant, K) penalises the model for missing
    relevant items that did NOT make it into the top-K — which the naive
    implementation (dividing by the number of *found* relevant items) fails to do,
    causing mAP to appear larger than P@K, which is mathematically impossible.
"""

import torch
import torch.nn.functional as F


def average_precision_at_k(
    relevant:       torch.Tensor,   # binary [N], sorted by descending similarity
    total_relevant: int,            # total ground-truth positives in the full gallery
    k:              int = None,     # truncate at k (None = use all)
) -> float:
    """
    Compute Average Precision for a single query using the standard IR formula.

        AP@K = (sum of precision-at-hit) / min(total_relevant, K)

    Args:
        relevant:       [N] binary tensor already sorted by descending similarity.
        total_relevant: Total number of ground-truth positives in the FULL gallery
                        (before any top-K truncation). Used as the denominator to
                        correctly penalise missing relevant items outside top-K.
        k:              Truncate ranked list at k. None = use entire list (mAP@all).

    Returns:
        AP as float in [0, 1].
    """
    if k is not None:
        relevant = relevant[:k]

    n_found = relevant.sum().item()
    if n_found == 0:
        return 0.0

    # Positions (1-indexed) of relevant items in the ranked list
    positions  = torch.where(relevant)[0].float() + 1.0       # [n_found]
    precisions = torch.arange(1, n_found + 1, dtype=torch.float32) / positions

    # Standard denominator: min(total ground-truth positives, K)
    # Ensures AP@K <= P@K always holds.
    denominator = min(total_relevant, k) if k is not None else total_relevant
    return precisions.sum().item() / denominator


def precision_at_k(relevant: torch.Tensor, k: int) -> float:
    """Precision@K for a single query."""
    return relevant[:k].float().mean().item()


@torch.no_grad()
def compute_retrieval_metrics(
    sk_feats:   torch.Tensor,   # [Nq, D] — query sketch features (normalised)
    ph_feats:   torch.Tensor,   # [Ng, D] — gallery photo features (normalised)
    sk_labels:  torch.Tensor,   # [Nq]    — query class indices
    ph_labels:  torch.Tensor,   # [Ng]    — gallery class indices
    map_k:      int  = None,    # truncation for mAP (None = mAP@all)
    prec_k:     int  = 100,     # K for P@K
) -> dict:
    """
    Compute mAP@{map_k} and P@{prec_k} for a full query/gallery set.

    Returns dict with keys: 'mAP', 'P@{prec_k}', 'map_k', 'prec_k'
    """
    sk_feats = F.normalize(sk_feats.float(), dim=-1)
    ph_feats = F.normalize(ph_feats.float(), dim=-1)

    # Similarity matrix [Nq, Ng]
    sim = sk_feats @ ph_feats.t()

    # Sort gallery by descending similarity for each query
    sorted_idx = sim.argsort(dim=-1, descending=True)   # [Nq, Ng]

    ap_list   = []
    prec_list = []

    for q in range(sk_feats.shape[0]):
        # Ground truth: gallery items with same class as query
        relevant_all   = (ph_labels == sk_labels[q]).long()   # [Ng]
        total_relevant = int(relevant_all.sum().item())

        # Reorder by similarity score
        relevant_sorted = relevant_all[sorted_idx[q]]          # [Ng]

        ap   = average_precision_at_k(relevant_sorted, total_relevant=total_relevant, k=map_k)
        prec = precision_at_k(relevant_sorted, k=prec_k)

        ap_list.append(ap)
        prec_list.append(prec)

    mAP = sum(ap_list)   / len(ap_list)
    P_K = sum(prec_list) / len(prec_list)

    return {
        'mAP':         mAP,
        f'P@{prec_k}': P_K,
        'map_k':       map_k,
        'prec_k':      prec_k,
    }


def get_metric_config(dataset: str) -> dict:
    """
    Return the standard evaluation metric configuration for each dataset.

    Per community protocol:
        sketchy_1 : mAP@all,  P@100
        sketchy_2 : mAP@200,  P@200
        tuberlin  : mAP@all,  P@100
        quickdraw : mAP@all,  P@200
    """
    cfg = {
        'sketchy_1': {'map_k': None, 'prec_k': 100},
        'sketchy_2': {'map_k': 200,  'prec_k': 200},
        'tuberlin':  {'map_k': None, 'prec_k': 100},
        'quickdraw': {'map_k': None, 'prec_k': 200},
    }
    return cfg.get(dataset, {'map_k': None, 'prec_k': 100})