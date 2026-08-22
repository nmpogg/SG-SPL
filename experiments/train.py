import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from experiments.options import parser
from src.model import SGSPLModel
from src.dataset_retrieval import TrainDataset, ValDataset
from src.splits import UNSEEN_CLASSES
from src.utils import CustomProgressBar


def main():
    opts = parser.parse_args()

    pl.seed_everything(opts.seed, workers=True)

    train_ds = TrainDataset(opts)

    seen_classes   = train_ds.seen_classes
    unseen_classes = UNSEEN_CLASSES[opts.dataset]

    # Nhãn toàn cục dùng chung cho mọi val set (seen + unseen không đè id lên nhau).
    all_classes = sorted(set(seen_classes) | set(unseen_classes))
    class_to_id = {c: i for i, c in enumerate(all_classes)}

    # ZS luôn cần: query = sketch unseen, gallery = photo unseen
    val_sk_ds = ValDataset(opts, unseen_classes, class_to_id, modality='sketch')
    val_ph_ds = ValDataset(opts, unseen_classes, class_to_id, modality='photo')

    seen_class_names = seen_classes

    train_loader = DataLoader(
        dataset = train_ds,
        batch_size = opts.batch_size,
        shuffle = True,
        num_workers = opts.num_workers
    )

    val_sk_loader = DataLoader(
        dataset = val_sk_ds,
        batch_size = opts.test_batch_size,
        shuffle = False,
        num_workers = opts.num_workers
    )
    val_ph_loader = DataLoader(
        dataset = val_ph_ds,
        batch_size = opts.test_batch_size,
        shuffle = False,
        num_workers = opts.num_workers
    )

    # Thứ tự loader = dataloader_idx trong model.validation_step:
    #   0 = sketch unseen (query),  1 = photo unseen (gallery)
    #   2 = sketch seen  (query),   3 = photo seen  (gallery)   ← chỉ khi bật 'gzs'
    val_loaders = [val_sk_loader, val_ph_loader]

    if 'gzs' in opts.eval:
        seen_sk_ds = ValDataset(opts, seen_classes, class_to_id, modality='sketch')
        seen_ph_ds = ValDataset(opts, seen_classes, class_to_id, modality='photo')

        # Sketch seen rất nhiều → lấy mẫu để đo gap nhanh (deterministic theo seed).
        if opts.gzs_seen_query_limit and len(seen_sk_ds) > opts.gzs_seen_query_limit:
            g = torch.Generator().manual_seed(opts.seed)
            keep = torch.randperm(len(seen_sk_ds), generator=g)[:opts.gzs_seen_query_limit].tolist()
            seen_sk_ds = torch.utils.data.Subset(seen_sk_ds, keep)

        val_loaders += [
            DataLoader(seen_sk_ds, batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers),
            DataLoader(seen_ph_ds, batch_size=opts.test_batch_size, shuffle=False, num_workers=opts.num_workers),
        ]

    model = SGSPLModel(opts, seen_class_names=seen_class_names)

    logger = TensorBoardLogger(save_dir=opts.log_dir)

    # Metric theo dõi để checkpoint / early-stop: ưu tiên ZS, nếu chỉ bật GZS thì theo GZS.
    monitor_metric = 'mAP' if 'zs' in opts.eval else 'GZS_mAP'

    checkpoint_cb = ModelCheckpoint(
        dirpath   = os.path.join(opts.ckpt_dir, opts.exp_name),
        filename  = '{epoch:02d}-{' + monitor_metric + ':.4f}',
        monitor   = monitor_metric,
        mode      = 'max',
        save_top_k = 1,
        save_last = True,
    )
    early_stop_cb = EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        mode='max',
        verbose=False,
    )
    prog_bar = CustomProgressBar()

    callbacks = [checkpoint_cb, early_stop_cb, prog_bar]

    trainer = pl.Trainer(
        min_epochs = 1,
        max_epochs = opts.max_epochs,
        benchmark = False,
        deterministic=True,
        logger = logger,
        accelerator = 'gpu',
        devices = opts.gpus,
        precision = opts.precision,
        callbacks = callbacks,
        check_val_every_n_epoch = opts.val_every,
        num_sanity_val_steps = opts.sanity_steps
    )

    if opts.ckpt_path:
        print(f"\n[INFO] Resuming training from: {opts.ckpt_path}\n")

    trainer.fit(
        model = model,
        train_dataloaders = train_loader,
        val_dataloaders = val_loaders,
        ckpt_path = opts.ckpt_path
    )

    print(f'\n✓ Training done. Best ZS-mAP: {model.best_zs_map:.4f}')
    print(f'  Best checkpoint: {checkpoint_cb.best_model_path}')

if __name__ == '__main__':
    main()
