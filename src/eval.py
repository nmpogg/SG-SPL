import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision, retrieval_precision


def average_precision_at_k(relevant: torch.Tensor, k: int = None) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        relevant: binary [N] tensor sorted by descending similarity,
                  1=relevant, 0=not relevant
        k:        truncate at k (None = use all)

    Returns:
        AP as float
    """
    if k is not None:
        relevant = relevant[:k]
    n = relevant.sum().item()
    if n == 0:
        return 0.0
    positions = torch.where(relevant)[0].float() + 1.0   # 1-indexed
    precisions = torch.arange(1, n + 1, dtype=torch.float32) / positions
    return precisions.mean().item()


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
    
    ap = torch.zeros(len(sk_feats))
    precision = torch.zeros(len(sk_feats))
    for idx, sk_feat in enumerate(sk_feats):
        cls = sk_labels[idx]
        distance = F.cosine_similarity(sk_feat.unsqueeze(0), ph_feats)
        target = torch.zeros(len(ph_feats), dtype=torch.bool, device=ph_feats.device)
        target[np.where(ph_labels == cls)] = True

        if map_k is not None:
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu(), top_k=map_k)
        else:
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        precision[idx] = retrieval_precision(distance.cpu(), target.cpu(), top_k=prec_k)
    
    mAP = torch.mean(ap)
    P_K = torch.mean(precision)

    if map_k is None:
        return {
            'mAP@all': mAP,
            f'P@{prec_k}': P_K,
            'map_k': map_k,
            'prec_k': prec_k,
        }
    else:
        return {
            f'mAP@{map_k}': mAP,
            f'P@{prec_k}': P_K,
            'map_k': map_k,
            'prec_k': prec_k,
        }


def get_metric_config(dataset: str) -> dict:
    """
    Return the standard evaluation metric configuration for each dataset.

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
