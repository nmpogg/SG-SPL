"""Evaluate a trained SG-SPL checkpoint with the training validation logic.

Use the same model options that were used for training, for example:

    python experiments/evaluate.py \
        --ckpt_path checkpoints/SG-SPL/last.ckpt \
        --dataset sketchy_2 \
        --root datasets/Sketchy/ \
        --split zs \
        --n_prompts 1

For cross-dataset evaluation, --dataset/--root/--split select the target
dataset. Add --train_dataset and --train_root for the source dataset used
to recover the checkpoint's seen class names. Model options must still
match those used for training.
"""

import os
import sys
from copy import deepcopy

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from experiments.options import parser
from src.dataset_retrieval import TrainDataset, ValDataset
from src.model import SGSPLModel
from src.utils import CustomProgressBar


def main():
    parser.add_argument(
        '--train_dataset', default=None,
        choices=['sketchy_1', 'sketchy_2', 'tuberlin', 'quickdraw'],
        help='Source training dataset for seen class names (default: --dataset)',
    )
    parser.add_argument(
        '--train_root', default=None,
        help='Source training data root for seen class names (default: --root)',
    )
    opts = parser.parse_args()

    if not opts.ckpt_path:
        parser.error('--ckpt_path is required for evaluation.')
    if not os.path.isfile(opts.ckpt_path):
        parser.error(f'Checkpoint not found: {opts.ckpt_path}')

    pl.seed_everything(opts.seed, workers=True)

    train_opts = deepcopy(opts)
    train_opts.dataset = opts.train_dataset or opts.dataset
    train_opts.root = opts.train_root or opts.root
    train_ds = TrainDataset(train_opts)
    val_sk_ds = ValDataset(opts, modality='sketch')
    val_ph_ds = ValDataset(opts, modality='photo')

    val_sk_loader = DataLoader(
        dataset=val_sk_ds,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )
    val_ph_loader = DataLoader(
        dataset=val_ph_ds,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )

    print(f'\n[INFO] Loading checkpoint for evaluation: {opts.ckpt_path}\n')
    model = SGSPLModel.load_from_checkpoint(
        checkpoint_path=opts.ckpt_path,
        opts=opts,
        seen_class_names=train_ds.seen_classes,
        strict=True,
        map_location='cpu',
    )

    trainer = pl.Trainer(
        benchmark=False,
        deterministic=True,
        logger=False,
        accelerator='gpu',
        devices=opts.gpus,
        precision=opts.precision,
        callbacks=[CustomProgressBar()],
    )

    results = trainer.validate(
        model=model,
        dataloaders=[val_sk_loader, val_ph_loader],
        verbose=False,
    )

    metrics = results[0] if results else {}
    if 'mAP' in metrics and 'precision' in metrics:
        print(
            f"\nEvaluation done. mAP: {metrics['mAP']:.3f}, "
            f"precision: {metrics['precision']:.3f}"
        )
    else:
        print('\nEvaluation fail!')


if __name__ == '__main__':
    main()
