import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from experiments.options import parser
from src.model import SGSPLModel
from src.dataset_retrieval import TrainDataset, ValDataset
from src.utils import CustomProgressBar


def main():
    opts = parser.parse_args()

    pl.seed_everything(opts.seed, workers=True)

    train_ds = TrainDataset(opts)
    val_sk_ds = ValDataset(opts, modality='sketch')
    val_ph_ds = ValDataset(opts, modality='photo')

    seen_class_names = train_ds.seen_classes

    train_loader = DataLoader(
        train_ds,
        batch_size   = opts.batch_size,
        shuffle      = True,
        num_workers  = opts.num_workers,
        pin_memory   = True,
        drop_last    = True,
    )

    val_sk_loader = DataLoader(
        val_sk_ds,
        batch_size  = opts.test_batch_size,
        shuffle     = False,
        num_workers = opts.num_workers,
        pin_memory  = True,
    )
    val_ph_loader = DataLoader(
        val_ph_ds,
        batch_size  = opts.test_batch_size,
        shuffle     = False,
        num_workers = opts.num_workers,
        pin_memory  = True,
    )

    model = SGSPLModel(opts, seen_class_names=seen_class_names)

    logger = TensorBoardLogger(save_dir=opts.log_dir)

    checkpoint_cb = ModelCheckpoint(
        dirpath   = os.path.join(opts.ckpt_dir, opts.exp_name),
        filename  = '{epoch:02d}-{mAP:.4f}',
        monitor   = 'mAP',
        mode      = 'max',
        save_top_k = 1,
        save_last = True,
    )
    early_stop_cb = EarlyStopping(
        monitor='mAP',
        patience=5,
        mode='max',
        verbose=False,
    )
    prog_bar = CustomProgressBar()

    callbacks = [checkpoint_cb, early_stop_cb, prog_bar]

    trainer = pl.Trainer(
        min_epochs = 1,
        max_epochs = opts.max_epochs,
        benchmark = True,
        logger = logger,
        callbacks = callbacks,
        check_val_every_n_epoch = opts.val_every,
        num_sanity_val_steps = opts.sanity_steps
    )

    if opts.ckpt_path:
        print(f"\n[INFO] Resuming training from: {opts.ckpt_path}\n")

    trainer.fit(
        model = model,
        train_dataloaders = train_loader,
        val_dataloaders = [val_sk_loader, val_ph_loader],
        ckpt_path = opts.ckpt_path
    )

    print(f'\n✓ Training done. Best ZS-mAP: {model.best_zs_map:.4f}')
    print(f'  Best checkpoint: {checkpoint_cb.best_model_path}')

if __name__ == '__main__':
    main()
