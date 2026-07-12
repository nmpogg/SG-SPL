"""
Standalone Evaluation Script
==============================
Re-evaluate a saved checkpoint with the CORRECTED mAP@K formula.

Usage:
    python experiments/evaluate.py \\
        --ckpt_path checkpoints/<exp_tag>/epoch005_mAP0.7100.ckpt \\
        --dataset sketchy_2 \\
        --sketchy_dir /path/to/Sketchy \\
        [--independent_ln]  # pass if the checkpoint was trained with this flag

Examples:
    # Evaluate a CLIP-AT baseline checkpoint
    python experiments/evaluate.py \\
        --ckpt_path checkpoints/clipat_sketchy_2_ssc0.0_x0.5_sph0.0-0.0_seed42/best.ckpt \\
        --dataset sketchy_2 \\
        --sketchy_dir datasets/Sketchy \\
        --independent_ln

    # Evaluate a SG-SPL checkpoint
    python experiments/evaluate.py \\
        --ckpt_path checkpoints/sgspl_sketchy_2_ssc1.0_x0.5_sph1.0-0.2_seed42/best.ckpt \\
        --dataset sketchy_2 \\
        --sketchy_dir datasets/Sketchy
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from src.model import SGSPLModel
from src.dataset_retrieval import get_dataset
from src.eval import compute_retrieval_metrics, get_metric_config
from experiments.options import parser as train_parser


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_eval_args():
    """Add eval-specific args to the existing train parser and parse."""
    train_parser.add_argument('--ckpt_path', type=str, required=True,
                              help='Path to the .ckpt file to evaluate.')
    train_parser.add_argument('--split', type=str, default='test', choices=['test', 'train'],
                              help='Which split to evaluate on (default: test).')
    return train_parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(opts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Build datasets ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading dataset: {opts.dataset} ({opts.split} split)")
    ds = get_dataset(opts, mode=opts.split)
    loader = DataLoader(
        ds,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
    )
    seen_class_names = get_dataset(opts, mode='train').seen_classes

    # ── Load model from checkpoint ─────────────────────────────────────────────
    print(f"\n[INFO] Loading checkpoint: {opts.ckpt_path}")
    model = SGSPLModel.load_from_checkpoint(
        opts.ckpt_path,
        opts=opts,
        seen_class_names=seen_class_names,
        strict=True,
    )
    model.eval().to(device)

    # ── Encode all samples ─────────────────────────────────────────────────────
    print(f"\n[INFO] Encoding {len(ds)} samples...")
    sk_feats_list  = []
    ph_feats_list  = []
    sk_labels_list = []
    ph_labels_list = []

    for batch in loader:
        sk, ph, _, _, cat_idx = batch
        sk = sk.to(device)
        ph = ph.to(device)

        sk_f = model(sk, modality='sketch').cpu()
        ph_f = model(ph, modality='image').cpu()

        sk_feats_list.append(sk_f)
        ph_feats_list.append(ph_f)
        sk_labels_list.append(cat_idx)
        ph_labels_list.append(cat_idx)

    sk_feats  = torch.cat(sk_feats_list)
    ph_feats  = torch.cat(ph_feats_list)
    sk_labels = torch.cat(sk_labels_list)
    ph_labels = torch.cat(ph_labels_list)

    # ── Compute ZS-SBIR metrics ────────────────────────────────────────────────
    metric_cfg = get_metric_config(opts.dataset)
    map_k  = metric_cfg['map_k']
    prec_k = metric_cfg['prec_k']
    print(f"\n[INFO] Computing ZS-SBIR metrics (mAP@{'all' if map_k is None else map_k}, P@{prec_k})...")

    zs_metrics = compute_retrieval_metrics(
        sk_feats  = sk_feats,
        ph_feats  = ph_feats,
        sk_labels = sk_labels,
        ph_labels = ph_labels,
        **metric_cfg,
    )

    print("\n" + "=" * 52)
    print(f"  CHECKPOINT : {os.path.basename(opts.ckpt_path)}")
    print(f"  DATASET    : {opts.dataset}")
    print(f"  ZS mAP@{'all' if map_k is None else map_k:<4}: {zs_metrics['mAP']:.4f}")
    print(f"  ZS P@{prec_k}    : {zs_metrics[f'P@{prec_k}']:.4f}")
    print("=" * 52 + "\n")


if __name__ == '__main__':
    opts = get_eval_args()
    run_evaluation(opts)
