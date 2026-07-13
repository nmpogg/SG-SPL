"""
Standalone Evaluation Script
==============================
Evaluate a saved checkpoint with the CORRECTED mAP@K formula and
a FULL deterministic gallery (no random sampling).

Usage:
    python experiments/evaluate.py \\
        --ckpt_path checkpoints/<exp_tag>/epoch010_mAP0.7100.ckpt \\
        --dataset sketchy_2 \\
        --sketchy_dir /path/to/Sketchy \\
        --n_prompts 3 \\
        [--independent_ln]

Note: Pass the same flags used during training so the model is
      reconstructed with the correct architecture.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from src.model import SGSPLModel
from src.dataset_retrieval import get_dataset, RetrievalEvalDataset
from src.eval import compute_retrieval_metrics, get_metric_config
from experiments.options import parser as train_parser


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_eval_args():
    """
    Parse args using the existing train_parser.
    --ckpt_path is already defined in experiments/options.py.
    """
    return train_parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build datasets ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading dataset: {opts.dataset}")
    # train_ds provides seen/unseen class splits for RetrievalEvalDataset
    train_ds = get_dataset(opts, mode='train')

    val_sk_ds = RetrievalEvalDataset(train_ds, modality='sketch')
    val_ph_ds = RetrievalEvalDataset(train_ds, modality='photo', include_seen=False)

    print(f"         Sketch queries : {len(val_sk_ds):,}")
    print(f"         Photo gallery  : {len(val_ph_ds):,}")

    val_sk_loader = DataLoader(
        val_sk_ds,
        batch_size  = opts.batch_size,
        shuffle     = False,
        num_workers = opts.num_workers,
        pin_memory  = True,
    )
    val_ph_loader = DataLoader(
        val_ph_ds,
        batch_size  = opts.batch_size,
        shuffle     = False,
        num_workers = opts.num_workers,
        pin_memory  = True,
    )

    seen_class_names = train_ds.seen_classes

    # ── Load model from checkpoint ─────────────────────────────────────────────
    print(f"\n[INFO] Loading checkpoint: {opts.ckpt_path}")
    model = SGSPLModel.load_from_checkpoint(
        opts.ckpt_path,
        opts=opts,
        seen_class_names=seen_class_names,
        strict=True,
    )
    model.eval().to(device)

    # ── Encode sketch queries ──────────────────────────────────────────────────
    print(f"\n[INFO] Encoding sketch queries ({len(val_sk_ds):,})...")
    sk_feats_list  = []
    sk_labels_list = []
    for imgs, cat_idx in val_sk_loader:
        imgs = imgs.to(device)
        sk_feats_list.append(model(imgs, modality='sketch').cpu())
        sk_labels_list.append(cat_idx)

    # ── Encode photo gallery ───────────────────────────────────────────────────
    print(f"[INFO] Encoding photo gallery ({len(val_ph_ds):,})...")
    ph_feats_list  = []
    ph_labels_list = []
    for imgs, cat_idx in val_ph_loader:
        imgs = imgs.to(device)
        ph_feats_list.append(model(imgs, modality='image').cpu())
        ph_labels_list.append(cat_idx)

    sk_feats  = torch.cat(sk_feats_list)
    ph_feats  = torch.cat(ph_feats_list)
    sk_labels = torch.cat(sk_labels_list)
    ph_labels = torch.cat(ph_labels_list)

    # ── Compute ZS-SBIR metrics ────────────────────────────────────────────────
    metric_cfg = get_metric_config(opts.dataset)
    map_k  = metric_cfg['map_k']
    prec_k = metric_cfg['prec_k']
    print(f"\n[INFO] Computing ZS-SBIR metrics "
          f"(mAP@{'all' if map_k is None else map_k}, P@{prec_k})...")

    zs_metrics = compute_retrieval_metrics(
        sk_feats  = sk_feats,
        ph_feats  = ph_feats,
        sk_labels = sk_labels,
        ph_labels = ph_labels,
        **metric_cfg,
    )

    print("\n" + "=" * 55)
    print(f"  CHECKPOINT : {os.path.basename(opts.ckpt_path)}")
    print(f"  DATASET    : {opts.dataset}")
    print(f"  QUERIES    : {len(val_sk_ds):,} sketches")
    print(f"  GALLERY    : {len(val_ph_ds):,} photos")
    print(f"  ZS mAP@{'all' if map_k is None else map_k:<4}: {zs_metrics['mAP']:.4f}")
    print(f"  ZS P@{prec_k}    : {zs_metrics[f'P@{prec_k}']:.4f}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    opts = get_eval_args()
    if not opts.ckpt_path:
        raise ValueError("Please provide --ckpt_path to the checkpoint you want to evaluate.")
    run_evaluation(opts)
