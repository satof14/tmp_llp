import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from collections import defaultdict
import random

from .base import BaseBagDataset, BaseSingleImageDataset


class HumanSomaticSmallBagDataset(BaseBagDataset):
    """Human Somatic Small dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', split='train', bag_size=5, transform=None, channel_stats=None):
        super().__init__(bag_size=bag_size, transform=transform)
        self.root = root
        self.split = split
        self.channel_stats = channel_stats
        
        # Load data
        self._load_data()
        
        # Set transform if not provided
        if self.transform is None:
            if self.channel_stats is None:
                raise ValueError("channel_stats must be provided when transform is None")
            self.transform = self._get_default_transform(is_train=(split == 'train'))
        
        # Create bags
        self.bags = self._create_bags()
    
    def _load_data(self):
        """Load Human Somatic Small dataset and organize by class."""
        # Initialize data containers
        self.data = []
        self.targets = []
        self.class_indices = defaultdict(list)
        
        # Load data from text files (train.txt, val.txt, test.txt)
        list_file = f'{self.split}.txt'
        with open(os.path.join(self.root, list_file), 'r') as f:
            for line in f:
                rel_path, label_idx = line.strip().split()
                img_path = os.path.join(self.root, self.split, rel_path)
                idx = len(self.data)
                self.data.append(img_path)
                self.targets.append(int(label_idx))
                self.class_indices[int(label_idx)].append(idx)
        
        self.num_classes = 3
        self.class_indices = dict(self.class_indices)
    
    def get_training_bags_indices(self):
        """Get the indices used in training bags for channel stats calculation."""
        train_bags = []
        for bag in self.bags:
            train_bags.append(bag['indices'])
        return train_bags
        
    def _get_default_transform(self, is_train):
        """Get default transform for Human Somatic Small."""
        mean = self.channel_stats['mean']
        std = self.channel_stats['std']
            
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(128, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    
    def _get_image(self, idx):
        """Get an image by index."""
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return img


class HumanSomaticSmallSingleImageDataset(BaseSingleImageDataset):
    """Human Somatic Small dataset for single image evaluation."""
    
    def __init__(self, root='./data', split='test', transform=None, channel_stats=None, max_samples=None):
        super().__init__(transform=transform)
        self.root = root
        self.split = split
        self.channel_stats = channel_stats
        self.max_samples = max_samples
        
        # Load data
        self._load_data()
        
        # Set transform if not provided
        if self.transform is None:
            if self.channel_stats is None:
                raise ValueError("channel_stats must be provided when transform is None")
            self.transform = self._get_default_transform()
    
    def _load_data(self):
        """Load Human Somatic Small dataset."""
        self.data = []
        self.targets = []
        
        # Load data from text files
        list_file = f'{self.split}.txt'
        with open(os.path.join(self.root, list_file), 'r') as f:
            lines = f.readlines()
        
        # Apply max_samples before processing if specified
        if self.max_samples is not None and len(lines) > self.max_samples:
            lines = random.sample(lines, self.max_samples)
        
        # Process selected lines
        for line in lines:
            rel_path, label_idx = line.strip().split()
            img_path = os.path.join(self.root, self.split, rel_path)
            self.data.append(img_path)
            self.targets.append(int(label_idx))
    
    def _get_default_transform(self):
        """Get default transform for Human Somatic Small."""
        mean = self.channel_stats['mean']
        std = self.channel_stats['std']
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def _get_image(self, idx):
        """Get an image by index."""
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        return img
    
    def _get_label(self, idx):
        """Get a label by index."""
        return self.targets[idx]
    
    def __len__(self):
        return len(self.data)


def get_human_somatic_small_bag_dataloader(root='./data', split='train', bag_size=5, batch_size=2, 
                                           num_workers=4, shuffle=True, channel_stats=None):
    """Get dataloader for Human Somatic Small bag-level training."""
    dataset = HumanSomaticSmallBagDataset(root=root, split=split, bag_size=bag_size, 
                                          channel_stats=channel_stats)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_human_somatic_small_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                                    num_workers=4, shuffle=False, channel_stats=None, 
                                                    max_samples=None):
    """Get dataloader for Human Somatic Small single image evaluation."""
    dataset = HumanSomaticSmallSingleImageDataset(root=root, split=split, 
                                                  channel_stats=channel_stats, 
                                                  max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader