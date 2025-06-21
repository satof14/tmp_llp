import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
from collections import defaultdict
import random

from .base import BaseBagDataset, BaseSingleImageDataset


class MIFCMBagDataset(BaseBagDataset):
    """MIFCM 3-classes dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', split='train', bag_size=5, transform=None, 
                 indices=None, channel_stats=None):
        super().__init__(bag_size=bag_size, transform=transform)
        self.root = root
        self.split = split
        self.indices = indices
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
        """Load MIFCM dataset and organize by class."""
        # Build path to dataset
        dataset_path = os.path.join(self.root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        
        # Initialize data containers
        self.data = []
        self.targets = []
        self.class_indices = defaultdict(list)
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data based on split
        if self.split in ['train', 'val']:
            # Load all data from train folder
            train_path = os.path.join(dataset_path, "train")
            all_data, all_targets = self._load_images_from_path(train_path, label_mapping)
            
            # For train split, use provided indices
            if self.split == 'train':
                if self.indices is None:
                    raise ValueError("indices must be provided for train split")
                self.data = [all_data[i] for i in self.indices]
                self.targets = [all_targets[i] for i in self.indices]
            else:  # split == 'val'
                # This should not be used for validation in the new setup
                raise ValueError("MIFCMBagDataset should not be used for validation split")
                
        else:  # split == 'test'
            # Load from test folder
            test_path = os.path.join(dataset_path, "test")
            self.data, self.targets = self._load_images_from_path(test_path, label_mapping)
        
        # Group by class
        for idx, target in enumerate(self.targets):
            self.class_indices[target].append(idx)
        
        self.num_classes = 3
        self.class_indices = dict(self.class_indices)
    
    def get_training_bags_indices(self):
        """Get the indices used in training bags for channel stats calculation."""
        train_bags = []
        for bag in self.bags:
            train_bags.append(bag['indices'])
        return train_bags
    
    def _load_images_from_path(self, path, label_mapping):
        """Load images and labels from directory structure."""
        data, targets = [], []
        for label_name in os.listdir(path):
            label_path = os.path.join(path, label_name)
            if os.path.isdir(label_path) and label_name in label_mapping:
                label_idx = label_mapping[label_name]
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        img_path = os.path.join(label_path, file)
                        data.append(img_path)
                        targets.append(label_idx)
        return data, targets
    
    def _get_default_transform(self, is_train):
        """Get default transform for MIFCM."""
        mean = self.channel_stats['mean']
        std = self.channel_stats['std']
            
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(64, padding=8),
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


class MIFCMSingleImageDataset(BaseSingleImageDataset):
    """MIFCM 3-classes dataset for single image evaluation."""
    
    def __init__(self, root='./data', split='test', transform=None, 
                 indices=None, channel_stats=None, max_samples=None):
        super().__init__(transform=transform)
        self.root = root
        self.split = split
        self.indices = indices
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
        """Load MIFCM dataset."""
        # Build path to dataset
        dataset_path = os.path.join(self.root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data based on split
        if self.split in ['train', 'val']:
            # Load all data from train folder
            train_path = os.path.join(dataset_path, "train")
            all_data, all_targets = self._load_images_from_path(train_path, label_mapping)
            
            if self.split in ['train', 'val']:
                if self.indices is None:
                    raise ValueError(f"indices must be provided for {self.split} split")
                # Apply max_samples to indices first
                indices_to_use = self.indices
                if self.max_samples is not None and len(indices_to_use) > self.max_samples:
                    indices_to_use = random.sample(indices_to_use, self.max_samples)
                self.data = [all_data[i] for i in indices_to_use]
                self.targets = [all_targets[i] for i in indices_to_use]
                
        else:  # split == 'test'
            # Load from test folder
            test_path = os.path.join(dataset_path, "test")
            self.data, self.targets = self._load_images_from_path(test_path, label_mapping)
    
    def _load_images_from_path(self, path, label_mapping):
        """Load images and labels from directory structure."""
        data, targets = [], []
        for label_name in os.listdir(path):
            label_path = os.path.join(path, label_name)
            if os.path.isdir(label_path) and label_name in label_mapping:
                label_idx = label_mapping[label_name]
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        img_path = os.path.join(label_path, file)
                        data.append(img_path)
                        targets.append(label_idx)
        return data, targets
    
    def _get_default_transform(self):
        """Get default transform for MIFCM."""
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


def get_mifcm_bag_dataloader(root='./data', split='train', bag_size=5, batch_size=2, 
                             num_workers=4, shuffle=True, indices=None, channel_stats=None):
    """Get dataloader for MIFCM bag-level training."""
    dataset = MIFCMBagDataset(root=root, split=split, bag_size=bag_size, 
                              indices=indices, channel_stats=channel_stats)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_mifcm_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                      num_workers=4, shuffle=False, indices=None, 
                                      channel_stats=None, max_samples=None):
    """Get dataloader for MIFCM single image evaluation."""
    dataset = MIFCMSingleImageDataset(root=root, split=split, indices=indices, 
                                      channel_stats=channel_stats, max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader