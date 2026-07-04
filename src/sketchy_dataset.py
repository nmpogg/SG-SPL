import os
import glob
import numpy as np
import torch
import random
from torchvision import transforms
from PIL import Image, ImageOps
from src.splits import UNSEEN_CLASSES, GENERALIZED_CLASSES, VISUALIZE_CLASSES
from src.utils import get_all_categories

def normal_transform(max_size=224):
    dataset_transforms = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, proportion=1.0):
        self.args = args
        self.proportion = proportion
        self.transform = normal_transform(self.args.max_size)
        
        unseen_classes = UNSEEN_CLASSES[self.args.dataset]

        self.all_categories = os.listdir(os.path.join(self.args.root, 'sketch'))
        self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
        
        if self.args.use_classes != 104:
            required_classes = list(GENERALIZED_CLASSES.get(self.args.dataset, []))
            other_classes = [c for c in self.all_categories if c not in required_classes]
            num_extra = max(0, self.args.use_classes - len(required_classes))
            extra_classes = random.sample(other_classes, num_extra)
            
            self.all_categories = sorted(required_classes + extra_classes)

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            sketch_paths = glob.glob(os.path.join(self.args.root, 'sketch', category, '*'))
            photo_paths = glob.glob(os.path.join(self.args.root, 'photo', category, '*'))
            
            if self.proportion != 1:
                num_sketch = max(1, int(len(sketch_paths) * self.proportion))
                num_photo  = max(1, int(len(photo_paths) * self.proportion))
                
                sketch_paths = random.sample(sketch_paths, num_sketch)
                photo_paths = random.sample(photo_paths, num_photo)
            
            self.all_sketches_path.extend(sketch_paths)
            self.all_photos_path[category] = photo_paths

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.args.max_size, self.args.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        return sk_tensor, img_tensor, neg_tensor, category, self.all_categories.index(category)


class ValidDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode='photo'):
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform(self.args.max_size)
        
        if self.args.visualize:
            self.unseen_classes = VISUALIZE_CLASSES[self.args.dataset]
        else:
            self.unseen_classes = UNSEEN_CLASSES[self.args.dataset]
            
        unseen_paths = []
        for category in self.unseen_classes:
            if self.mode == 'photo':
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
            else:
                unseen_paths.extend(glob.glob(os.path.join(self.args.root, 'sketch', category, '*')))

        self.paths = list(unseen_paths)

        if self.mode == 'photo':
            if getattr(self.args, 'gzs', False):
                seen_classes = get_all_categories(self.args, mode="train")
                for category in seen_classes:
                    self.paths.extend(glob.glob(os.path.join(self.args.root, 'photo', category, '*')))
            
    def __getitem__(self, index):
        filepath = self.paths[index]                
        category = filepath.split(os.path.sep)[-2]
        
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),  size=(self.args.max_size, self.args.max_size))
        image_tensor = self.transform(image)
        
        try:
            label = self.unseen_classes.index(category)
        except ValueError:
            label = -1
        
        return image_tensor, category
    
    def __len__(self):
        return len(self.paths)