import os
import glob
import numpy as np
from PIL import Image, ImageOps

from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from src.splits import UNSEEN_CLASSES, GENERALIZED_CLASSES

def normal_transform(image_size: int = 224):
    dataset_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return dataset_transforms

class TrainDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        self.transform = normal_transform(opts.image_size)

        all_classes = os.listdir(os.path.join(self.opts.root, 'sketch'))
        unseen_classes = UNSEEN_CLASSES[self.opts.dataset]
        self.seen_classes = sorted(list(set(all_classes) - set(unseen_classes)))

        self.all_sketches_path = []
        self.all_photos_path = {}

        for cls in self.seen_classes:
            sketch_paths = glob.glob(os.path.join(self.opts.root, 'sketch', cls, '*'))
            photo_paths = glob.glob(os.path.join(self.opts.root, 'photo', cls, '*'))

            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[cls] = photo_paths

    def __len__(self):
        return len(self.all_sketches_path)

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]                
        cls = sk_path.split(os.path.sep)[-2]

        neg_classes = self.seen_classes.copy()
        neg_classes.remove(cls)

        pos_path = np.random.choice(self.all_photos_path[cls])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
        pos_data = ImageOps.pad(Image.open(pos_path).convert('RGB'), size=(self.opts.image_size, self.opts.image_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.image_size, self.opts.image_size))

        sk_tensor  = self.transform(sk_data)
        pos_tensor = self.transform(pos_data)
        neg_tensor = self.transform(neg_data)
        
        return sk_tensor, pos_tensor, neg_tensor, self.seen_classes.index(cls)


class UniqueClassBatchSampler(Sampler):
    """Use every sketch once per epoch, with distinct classes in each batch."""

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indices_by_class = {}
        for index, path in enumerate(dataset.all_sketches_path):
            cls = os.path.basename(os.path.dirname(path))
            self.indices_by_class.setdefault(cls, []).append(index)

        if not 1 <= batch_size <= len(self.indices_by_class):
            raise ValueError(
                'batch_size must be between 1 and the number of non-empty seen classes '
                f'({len(self.indices_by_class)}) for unique-class batches.'
            )

    def __iter__(self):
        queues = {
            cls: np.random.permutation(indices).tolist()
            for cls, indices in self.indices_by_class.items()
        }
        active_classes = list(queues)
        while active_classes:
            # Prioritize longer queues to minimize small batches; shuffle ties.
            np.random.shuffle(active_classes)
            active_classes.sort(key=lambda cls: len(queues[cls]), reverse=True)
            batch = [queues[cls].pop() for cls in active_classes[:self.batch_size]]
            np.random.shuffle(batch)
            yield batch
            active_classes = [cls for cls in active_classes if queues[cls]]

    def __len__(self):
        counts = [len(indices) for indices in self.indices_by_class.values()]
        return max(max(counts), (sum(counts) + self.batch_size - 1) // self.batch_size)


class ValDataset(Dataset):
   
    def __init__(self, opts, modality = 'photo'):
        super().__init__()
        self.transform = normal_transform(opts.image_size)
        self.modality  = modality
        self.opts = opts
        self.seed = 42

        self.unseen_classes = UNSEEN_CLASSES[self.opts.dataset]

        if self.opts.split == 'gzs':
            self.generalized_classes = GENERALIZED_CLASSES[self.opts.dataset]
        else:
            self.generalized_classes = []

        self.val_classes = self.unseen_classes + self.generalized_classes

        unseen_paths = []
        for cls in self.unseen_classes:
            if self.modality == 'photo':
                unseen_paths.extend(glob.glob(os.path.join(self.opts.root, 'photo', cls, '*')))
            else:
                unseen_paths.extend(glob.glob(os.path.join(self.opts.root, 'sketch', cls, '*')))

        self.paths = list(unseen_paths)

        if self.modality == 'photo':
            if self.opts.split == 'gzs':
                for cls in self.generalized_classes:
                    self.paths.extend(glob.glob(os.path.join(self.opts.root, 'photo', cls, '*')))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        filepath = self.paths[index]                
        cls = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
        image_tensor = self.transform(image)

        return image_tensor, self.val_classes.index(cls)

