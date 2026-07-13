"""
Dataset classes for Zero-Shot Sketch-Based Image Retrieval (ZS-SBIR).

Supports: Sketchy-Extended, TU-Berlin-Extended, QuickDraw-Extended.

Training dataset (BaseRetrievalDataset, mode='train'):
    Each __getitem__ returns:
        (sk_tensor, ph_tensor, neg_ph_tensor, cat_name, cat_idx)

Evaluation dataset (RetrievalEvalDataset):
    Each __getitem__ returns:
        (image_tensor, cat_idx)
    Covers ALL sketches or ALL gallery photos without random sampling.
    Use two separate DataLoaders (sketch + photo) for correct retrieval eval.

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
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.splits import UNSEEN_CLASSES

def normal_transform(image_size: int = 224):
    dataset_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

_IMG_EXTS = ('*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.PNG', '*.JPG')


def collect_files(directory: str) -> list:
    """Collect all image files under a directory."""
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


class BaseRetrievalDataset(Dataset):
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
        self.opts = opts
        self.mode = mode
        self.transform = transform or normal_transform(opts.image_size)
        self.build_index()

    # to be set by subclass
    sketch_dir: str = ''
    photo_dir:  str = ''
    seen_classes:   list = []
    unseen_classes: list = []

    # ──────────────────────────────────────────────────────────────────────
    def build_index(self):
        """Index all sketch/photo files for the current mode."""
        if self.mode == 'train':
            classes = self.seen_classes
        else:
            classes = self.unseen_classes

        self.classes = sorted(classes)
        self.class2idx = {c: i for i, c in enumerate(self.classes)}

        # {class_name: [file_paths]}
        self.sk_files = {}
        self.ph_files = {}

        for cls in self.classes:
            sk_dir = os.path.join(self.sketch_dir, cls)
            ph_dir = os.path.join(self.photo_dir,  cls)
            sk = collect_files(sk_dir)
            ph = collect_files(ph_dir)
            if sk and ph:
                self.sk_files[cls] = sk
                self.ph_files[cls] = ph

        self.classes = [c for c in self.classes if c in self.sk_files]

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

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
        ph_data = ImageOps.pad(Image.open(ph_path).convert('RGB'), size=(self.opts.image_size, self.opts.image_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.image_size, self.opts.image_size))

        sk_tensor  = self.transform(sk_data)
        ph_tensor  = self.transform(ph_data)
        neg_tensor = self.transform(neg_data)

        return sk_tensor, ph_tensor, neg_tensor, cat_name, torch.tensor(cat_idx, dtype=torch.long)

    # test helpers for retrieval evaluation
    def get_all_sketches(self):
        """Returns (tensor_stack, cat_names, cat_indices) for all test sketches."""
        imgs, names, idxs = [], [], []
        for sk_path, cat in self.items:
            sk_data = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
            imgs.append(self.transform(sk_data))
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
            for ph_path in collect_files(ph_dir):
                ph_data = ImageOps.pad(Image.open(ph_path).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
                imgs.append(self.transform(ph_data))
                names.append(cls)
                idxs.append(all_class2idx.get(cls, -1))

        return torch.stack(imgs), names, torch.tensor(idxs)

class SketchyDataset(BaseRetrievalDataset):
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

        self.sketch_dir = os.path.join(data_dir, 'sketch')
        self.photo_dir = os.path.join(data_dir, 'photo')
        self.seen_classes = seen
        self.unseen_classes = unseen

        super().__init__(opts, mode, transform)


class TUBerlinDataset(BaseRetrievalDataset):
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

class QuickDrawDataset(BaseRetrievalDataset):
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


DATASET_MAP = {
    'sketchy_1': SketchyDataset,
    'sketchy_2': SketchyDataset,
    'tuberlin':  TUBerlinDataset,
    'quickdraw': QuickDrawDataset,
}


class RetrievalEvalDataset(Dataset):
    """
    Flat dataset for retrieval evaluation.

    Returns individual images (sketch OR photo) with their class index.
    Covers ALL files deterministically — no random sampling.

    Usage:
        base_ds = get_dataset(opts, mode='train')   # to get seen/unseen splits
        sk_ds = RetrievalEvalDataset(base_ds, modality='sketch')
        ph_ds = RetrievalEvalDataset(base_ds, modality='photo', include_seen=False)

    Args:
        base_ds:      An instance of BaseRetrievalDataset (any subclass).
        modality:     'sketch' | 'photo'
        include_seen: If True, gallery includes seen-class photos (for GZS-SBIR).
                      Only applies when modality='photo'.
    """

    def __init__(
        self,
        base_ds:      BaseRetrievalDataset,
        modality:     str  = 'photo',
        include_seen: bool = False,
    ):
        super().__init__()
        self.transform = base_ds.transform
        self.modality  = modality

        # Build a global class index spanning seen + unseen
        all_classes    = sorted(set(base_ds.seen_classes) | set(base_ds.unseen_classes))
        self.class2idx = {c: i for i, c in enumerate(all_classes)}

        self.items = []   # list of (file_path, cat_idx)

        if modality == 'sketch':
            # All test (unseen-class) sketches
            for cls in sorted(base_ds.unseen_classes):
                sk_dir = os.path.join(base_ds.sketch_dir, cls)
                for fp in collect_files(sk_dir):
                    self.items.append((fp, self.class2idx[cls]))

        elif modality == 'photo':
            # Gallery photos: unseen classes (+ seen if GZS-SBIR)
            gallery_classes = list(base_ds.unseen_classes)
            if include_seen:
                gallery_classes += list(base_ds.seen_classes)
            for cls in sorted(gallery_classes):
                ph_dir = os.path.join(base_ds.photo_dir, cls)
                for fp in collect_files(ph_dir):
                    self.items.append((fp, self.class2idx[cls]))
        else:
            raise ValueError(f"modality must be 'sketch' or 'photo', got '{modality}'")

        if len(self.items) == 0:
            raise RuntimeError(
                f'RetrievalEvalDataset: no files found for modality={modality}. '
                f'Check data directories.'
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        fp, cat_idx = self.items[index]
        img_size = self.transform.transforms[0].size
        # size may be int or (h, w)
        sz = img_size if isinstance(img_size, int) else img_size[0]
        img = ImageOps.pad(Image.open(fp).convert('RGB'), size=(sz, sz))
        return self.transform(img), torch.tensor(cat_idx, dtype=torch.long)



def get_dataset(opts, mode: str = 'train', transform=None):
    """
    Factory function: returns the correct Dataset for opts.dataset.

    Usage:
        train_ds = get_dataset(opts, mode='train')
        test_ds  = get_dataset(opts, mode='test')
    """
    cls = DATASET_MAP.get(opts.dataset)
    if cls is None:
        raise ValueError(f'Unknown dataset: {opts.dataset}. Choose from {list(DATASET_MAP)}')
    return cls(opts, mode=mode, transform=transform)
