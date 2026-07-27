import os
import glob
import numpy as np
from PIL import Image, ImageOps

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


class ValDataset(Dataset):
   
    def __init__(self, opts, modality = 'photo'):
        super().__init__()
        self.transform = normal_transform(opts.image_size)
        self.modality  = modality
        self.opts = opts
        self.seed = 42

        self.unseen_classes = UNSEEN_CLASSES[opts.dataset]

        unseen_paths = []
        for cls in self.unseen_classes:
            if self.modality == 'photo':
                unseen_paths.extend(glob.glob(os.path.join(self.opts.root, 'photo', cls, '*')))
            else:
                unseen_paths.extend(glob.glob(os.path.join(self.opts.root, 'sketch', cls, '*')))

        self.paths = list(unseen_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        filepath = self.paths[index]                
        cls = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.opts.image_size, self.opts.image_size))
        image_tensor = self.transform(image)
        
        return image_tensor, self.unseen_classes.index(cls)

