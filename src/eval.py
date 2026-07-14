import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.retrieval import retrieval_average_precision

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
