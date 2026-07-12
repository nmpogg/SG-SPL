"""
SG-SPL Training Script
======================
Entry point for training with PyTorch Lightning 2.6.4.

Usage:
    python experiments/train.py \
        --dataset sketchy_2 \
        --sketchy_dir datasets/Sketchy \
        --n_prompts 16 \
        --l_cls 1.0 \
        --l_ssc 1.0 \
        --l_x 0.5 \
        --l_sph_ph 1.0 \
        --l_sph_sk 0.2 \
        --max_epochs 100

Ablation examples:
    # Baseline (triplet + cls only)
    python experiments/train.py --l_ssc 0 --l_sph_ph 0 --l_sph_sk 0

    # + L_SSC only (no xmod)
    python experiments/train.py --l_ssc 1.0 --l_x 0

    # + L_SSC + L_xmod
    python experiments/train.py --l_ssc 1.0 --l_x 0.5 --l_sph_ph 0 --l_sph_sk 0

    # Full SG-SPL
    python experiments/train.py --l_ssc 1.0 --l_x 0.5 --l_sph_ph 1.0 --l_sph_sk 0.2
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# os.environ['FORCE_SIMPLE_PROGRESSBAR'] = '1'

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.options import parser          # argparse parser
from src.model import SGSPLModel
from src.dataset_retrieval import get_dataset
from src.utils import CustomProgressBar


def main():
    opts = parser.parse_args()

    pl.seed_everything(opts.seed, workers=True)

    train_ds = get_dataset(opts, mode='train')
    val_ds   = get_dataset(opts, mode='test')

    seen_class_names = train_ds.seen_classes       # used to build text anchor

    train_loader = DataLoader(
        train_ds,
        batch_size   = opts.batch_size,
        shuffle      = True,
        num_workers  = opts.num_workers,
        pin_memory   = True,
        drop_last    = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = opts.batch_size,
        shuffle      = False,
        num_workers  = opts.num_workers,
        pin_memory   = True,
    )

    model = SGSPLModel(opts, seen_class_names=seen_class_names)

    exp_tag = (
        f'{opts.exp_name}_'
        f'{opts.dataset}_'
        f'ssc{opts.l_ssc}_x{opts.l_x}_'
        f'sph{opts.l_sph_ph}-{opts.l_sph_sk}_'
        f'seed{opts.seed}'
    )
    logger = TensorBoardLogger(save_dir=opts.log_dir, name=exp_tag)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = os.path.join(opts.ckpt_dir, exp_tag),
        filename  = 'epoch{epoch:03d}_mAP{mAP:.4f}',
        monitor   = 'mAP',
        mode      = 'max',
        save_top_k = 1,
        save_last = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_cb = EarlyStopping(
        monitor='mAP',
        patience=5,
        mode='max',
        verbose=False,
    )
    prog_bar = CustomProgressBar(refresh_rate=1)

    callbacks = [checkpoint_cb, lr_monitor, early_stop_cb, prog_bar]

    trainer = pl.Trainer(
        max_epochs         = opts.max_epochs,
        logger             = logger,
        callbacks          = callbacks,
        accelerator        = 'gpu' if torch.cuda.is_available() else 'cpu',
        devices            = opts.gpus,
        precision          = opts.precision,     # '16-mixed' or '32'
        gradient_clip_val  = opts.grad_clip,
        check_val_every_n_epoch = opts.val_every,
        num_sanity_val_steps = opts.sanity_steps,
        log_every_n_steps  = 10,
        deterministic      = False,
        enable_progress_bar = True,             # keep tqdm; suppress Rich
    )

    # --- Auto Resume Logic ---
    ckpt_path = opts.ckpt_path
    if not ckpt_path:
        auto_ckpt_path = os.path.join(opts.ckpt_dir, exp_tag, 'last.ckpt')
        if os.path.exists(auto_ckpt_path):
            ckpt_path = auto_ckpt_path

    if ckpt_path:
        print(f"\n[INFO] Resuming training from: {ckpt_path}\n")

    trainer.validate(
        model = model,
        dataloaders = val_loader,
        ckpt_path = ckpt_path
    )

    # print(f'\n✓ Training done. Best ZS-mAP: {model.best_zs_map:.4f}')
    # print(f'  Best checkpoint: {checkpoint_cb.best_model_path}')

if __name__ == '__main__':
    main()
