import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

def retrieval_precision(preds, target, top_k):
    sorted_idx = preds.argsort(dim=-1, descending=True)
    sorted_target = target[sorted_idx]

    tot_pos = sorted_target.sum().item()

    if tot_pos == 0:
        return torch.tensor(0.0, device=preds.device)

    if top_k is not None:
        top = min(top_k, int(tot_pos))
    else:
        top = int(tot_pos)

    return sorted_target[:top].float().mean()

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

    return {
        'mAP': mAP,
        'precision': P_K,
        'map_k': map_k,
        'prec_k': prec_k,
    }


@torch.no_grad()
def compute_retrieval_metrics_batched(
    sk_feats: torch.Tensor,
    ph_feats: torch.Tensor,
    sk_labels: torch.Tensor,
    ph_labels: torch.Tensor,
    map_k: int = None,
    prec_k: int = 100,
    batch_size: int = 256,
) -> dict:
    """Evaluate several queries at once without storing the full query-gallery matrix."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    # The existing metric uses cosine_similarity even for normalized embeddings.
    sk_feats = F.normalize(sk_feats.float(), dim=-1, eps=1e-8)
    ph_feats = F.normalize(ph_feats.float(), dim=-1, eps=1e-8)
    sk_labels = sk_labels.to(sk_feats.device)
    ph_labels = ph_labels.to(sk_feats.device)
    gallery_size = len(ph_feats)
    rank = torch.arange(1, gallery_size + 1, device=sk_feats.device)
    ap_total = torch.zeros((), device=sk_feats.device)
    precision_total = torch.zeros((), device=sk_feats.device)

    for start in range(0, len(sk_feats), batch_size):
        scores = sk_feats[start:start + batch_size] @ ph_feats.T
        target = sk_labels[start:start + batch_size, None] == ph_labels[None, :]

        # Match torchmetrics.retrieval_average_precision: top-k ranking,
        # ignore nonpositive scores, then average precision over retrieved hits.
        ap_scores, ap_indices = scores.topk(
            min(map_k or gallery_size, gallery_size), dim=1, sorted=True
        )
        ap_hits = (target.gather(1, ap_indices) & (ap_scores > 0)).float()
        ap_rank = rank[:ap_hits.shape[1]]
        ap = ((ap_hits.cumsum(1) / ap_rank) * ap_hits).sum(1)
        ap_total += (ap / ap_hits.sum(1).clamp_min(1)).sum()

        # The original P@K uses min(K, total positives), including positives
        # outside the first K results when choosing the denominator.
        ranked_target = target.gather(1, scores.argsort(dim=1, descending=True))
        positive_count = target.sum(1)
        top = positive_count.clamp(max=prec_k) if prec_k is not None else positive_count
        precision_hits = (ranked_target & (rank[None, :] <= top[:, None])).sum(1)
        precision_total += (precision_hits / top.clamp_min(1)).sum()

    n_queries = len(sk_feats)
    return {
        'mAP': (ap_total / n_queries).cpu(),
        'precision': (precision_total / n_queries).cpu(),
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
