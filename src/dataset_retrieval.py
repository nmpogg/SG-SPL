"""
Dataset classes for Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR).

Supports: Sketchy-Extended, TU-Berlin-Extended, QuickDraw-Extended.

Each __getitem__ returns:
    (sk_tensor, ph_tensor, neg_ph_tensor, cat_name, cat_idx)
where cat_idx is the integer index into the seen/all class list —
required for L_cls and the prototype bank.

Data directory structure expected:
    Sketchy/
        sketch/<class_name>/*.png
        photo/<class_name>/*.jpg
    TUBerlin/
        sketches/<class_name>/*.png
        images/<class_name>/*.jpg
    QuickDraw/
        sketches/<class_name>/*.png
        images/<class_name>/*.jpg
"""

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.splits import UNSEEN_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transform(image_size: int = 224, mode: str = 'train'):
    """Standard CLIP-compatible image transform."""
    normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std= (0.26862954, 0.26130258, 0.27577711),
    )
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:   # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.PNG', '*.JPG')


def _collect_files(directory: str) -> list:
    """Collect all image files under a directory."""
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


# ─────────────────────────────────────────────────────────────────────────────
# Base dataset (triplet: sketch + positive photo + negative photo)
# ─────────────────────────────────────────────────────────────────────────────

class _BaseRetrievalDataset(Dataset):
    """
    Abstract base for sketch-photo retrieval datasets.
    Subclasses must set:
        self.sketch_dir : root of sketch folders
        self.photo_dir  : root of photo folders
        self.seen_classes   : list of seen class names (train)
        self.unseen_classes : list of unseen class names (test)
    """

    def __init__(self, opts, mode: str = 'train', transform=None):
        """
        Args:
            opts:      argparse namespace from experiments/options.py
            mode:      'train' | 'val' | 'test'
            transform: torchvision transform (defaults to get_transform)
        """
        self.opts      = opts
        self.mode      = mode
        self.transform = transform or get_transform(opts.image_size, mode)
        self._build_index()

    # ── to be set by subclass ──────────────────────────────────────────────
    sketch_dir: str = ''
    photo_dir:  str = ''
    seen_classes:   list = []
    unseen_classes: list = []

    # ──────────────────────────────────────────────────────────────────────
    def _build_index(self):
        """Index all sketch/photo files for the current mode."""
        if self.mode == 'train':
            classes = self.seen_classes
        else:
            classes = self.unseen_classes

        self.classes    = sorted(classes)
        self.class2idx  = {c: i for i, c in enumerate(self.classes)}

        # {class_name: [file_paths]}
        self.sk_files  = {}
        self.ph_files  = {}

        for cls in self.classes:
            sk_dir = os.path.join(self.sketch_dir, cls)
            ph_dir = os.path.join(self.photo_dir,  cls)
            sk = _collect_files(sk_dir)
            ph = _collect_files(ph_dir)
            if sk and ph:
                self.sk_files[cls] = sk
                self.ph_files[cls] = ph

        self.classes = [c for c in self.classes if c in self.sk_files]

        # Flat list for __len__ — one entry per sketch
        self.items = []   # list of (sketch_path, class_name)
        for cls in self.classes:
            for sk_path in self.sk_files[cls]:
                self.items.append((sk_path, cls))

        if len(self.items) == 0:
            raise RuntimeError(
                f'No data found for mode={self.mode}. '
                f'Check data directories:\n  sketch: {self.sketch_dir}\n  photo:  {self.photo_dir}'
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sk_path, cat_name = self.items[index]
        cat_idx = self.class2idx[cat_name]

        # Positive photo (same class)
        ph_path = random.choice(self.ph_files[cat_name])

        # Negative photo (different class)
        neg_cls = random.choice([c for c in self.classes if c != cat_name])
        neg_path = random.choice(self.ph_files[neg_cls])

        sk  = self.transform(_load_rgb(sk_path))
        ph  = self.transform(_load_rgb(ph_path))
        neg = self.transform(_load_rgb(neg_path))

        return sk, ph, neg, cat_name, torch.tensor(cat_idx, dtype=torch.long)

    # ── val / test helpers for retrieval evaluation ────────────────────────

    def get_all_sketches(self):
        """Returns (tensor_stack, cat_names, cat_indices) for all test sketches."""
        imgs, names, idxs = [], [], []
        for sk_path, cat in self.items:
            imgs.append(self.transform(_load_rgb(sk_path)))
            names.append(cat)
            idxs.append(self.class2idx[cat])
        return torch.stack(imgs), names, torch.tensor(idxs)

    def get_all_photos(self, include_seen: bool = False):
        """
        Returns (tensor_stack, cat_names, cat_indices) for gallery photos.

        ZS-SBIR:   include_seen=False  → gallery = unseen class photos only
        GZS-SBIR:  include_seen=True   → gallery = seen + unseen class photos
        """
        gallery_classes = self.unseen_classes
        if include_seen:
            gallery_classes = self.seen_classes + self.unseen_classes

        # Rebuild photo index for gallery classes
        imgs, names, idxs = [], [], []
        all_classes = sorted(set(self.seen_classes + self.unseen_classes))
        all_class2idx = {c: i for i, c in enumerate(all_classes)}

        for cls in gallery_classes:
            ph_dir = os.path.join(self.photo_dir, cls)
            for ph_path in _collect_files(ph_dir):
                imgs.append(self.transform(_load_rgb(ph_path)))
                names.append(cls)
                idxs.append(all_class2idx.get(cls, -1))

        return torch.stack(imgs), names, torch.tensor(idxs)


# ─────────────────────────────────────────────────────────────────────────────
# Sketchy-Extended
# ─────────────────────────────────────────────────────────────────────────────

class SketchyDataset(_BaseRetrievalDataset):
    """
    Sketchy-Extended dataset.
    Supports split-1 (100/25 random) and split-2 (104/21 non-ImageNet).
    """

    def __init__(self, opts, mode='train', transform=None):
        split_key = opts.dataset   # 'sketchy_1' or 'sketchy_2'
        data_dir  = opts.sketchy_dir

        unseen = UNSEEN_CLASSES[split_key]
        # Seen = all categories in the sketch folder minus unseen
        all_cats = sorted([
            d for d in os.listdir(os.path.join(data_dir, 'sketch'))
            if os.path.isdir(os.path.join(data_dir, 'sketch', d))
        ])
        seen = [c for c in all_cats if c not in unseen]

        self.sketch_dir     = os.path.join(data_dir, 'sketch')
        self.photo_dir      = os.path.join(data_dir, 'photo')
        self.seen_classes   = seen
        self.unseen_classes = unseen

        super().__init__(opts, mode, transform)


# ─────────────────────────────────────────────────────────────────────────────
# TU-Berlin-Extended
# ─────────────────────────────────────────────────────────────────────────────

class TUBerlinDataset(_BaseRetrievalDataset):
    """TU-Berlin-Extended dataset (220 seen / 30 unseen)."""

    def __init__(self, opts, mode='train', transform=None):
        data_dir = opts.tuberlin_dir
        unseen   = UNSEEN_CLASSES['tuberlin']

        all_cats = sorted([
            d for d in os.listdir(os.path.join(data_dir, 'sketches'))
            if os.path.isdir(os.path.join(data_dir, 'sketches', d))
        ])
        seen = [c for c in all_cats if c not in unseen]

        self.sketch_dir     = os.path.join(data_dir, 'sketches')
        self.photo_dir      = os.path.join(data_dir, 'images')
        self.seen_classes   = seen
        self.unseen_classes = unseen

        super().__init__(opts, mode, transform)


# ─────────────────────────────────────────────────────────────────────────────
# QuickDraw-Extended
# ─────────────────────────────────────────────────────────────────────────────

class QuickDrawDataset(_BaseRetrievalDataset):
    """QuickDraw-Extended dataset (80 seen / 30 unseen)."""

    def __init__(self, opts, mode='train', transform=None):
        data_dir = opts.quickdraw_dir
        unseen   = UNSEEN_CLASSES['quickdraw']

        all_cats = sorted([
            d for d in os.listdir(os.path.join(data_dir, 'sketches'))
            if os.path.isdir(os.path.join(data_dir, 'sketches', d))
        ])
        seen = [c for c in all_cats if c not in unseen]

        self.sketch_dir     = os.path.join(data_dir, 'sketches')
        self.photo_dir      = os.path.join(data_dir, 'images')
        self.seen_classes   = seen
        self.unseen_classes = unseen

        super().__init__(opts, mode, transform)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_MAP = {
    'sketchy_1': SketchyDataset,
    'sketchy_2': SketchyDataset,
    'tuberlin':  TUBerlinDataset,
    'quickdraw': QuickDrawDataset,
}


def get_dataset(opts, mode: str = 'train', transform=None):
    """
    Factory function: returns the correct Dataset for opts.dataset.

    Usage:
        train_ds = get_dataset(opts, mode='train')
        test_ds  = get_dataset(opts, mode='test')
    """
    cls = _DATASET_MAP.get(opts.dataset)
    if cls is None:
        raise ValueError(f'Unknown dataset: {opts.dataset}. Choose from {list(_DATASET_MAP)}')
    return cls(opts, mode=mode, transform=transform)
