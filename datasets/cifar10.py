import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import random

from .base import BaseBagDataset, BaseSingleImageDataset


class CIFAR10BagDataset(BaseBagDataset):
    """CIFAR-10 dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', train=True, bag_size=5, transform=None, download=True, 
                 indices=None):
        super().__init__(bag_size=bag_size, transform=transform)
        self.root = root
        self.train = train
        self.download = download
        self.indices = indices
        
        # Load data
        self._load_data()
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform(train)
        
        # Create bags
        self.bags = self._create_bags()
    
    def _load_data(self):
        """Load CIFAR-10 dataset and organize by class."""
        # Load CIFAR-10 dataset
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download
        )
        
        # Filter indices based on provided indices
        if self.indices is not None:
            # Use only provided indices for bag creation
            self.available_indices = set(self.indices)
        else:
            # Use all indices (when no indices are provided)
            self.available_indices = set(range(len(self.cifar10)))
        
        # Group indices by class (only available indices)
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.cifar10):
            if idx in self.available_indices:
                self.class_indices[label].append(idx)
                
        # Convert to dict for consistency
        self.class_indices = dict(self.class_indices)
        self.num_classes = 10  # CIFAR-10 has 10 classes
    
    def _get_default_transform(self, is_train):
        """Get default transform for CIFAR-10."""
        if is_train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
    
    def _get_image(self, idx):
        """Get an image by index."""
        img, _ = self.cifar10[idx]
        return img


class CIFAR10SingleImageDataset(BaseSingleImageDataset):
    """CIFAR-10 dataset for single image evaluation."""
    
    def __init__(self, root='./data', split='test', transform=None, download=True, 
                 indices=None, max_samples=None):
        super().__init__(transform=transform)
        self.root = root
        self.split = split
        self.download = download
        self.indices = indices
        self.max_samples = max_samples
        
        # Load data
        self._load_data()
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _load_data(self):
        """Load CIFAR-10 dataset."""
        # For train/val splits, indices must be provided
        if self.split in ['train', 'val'] and self.indices is None:
            raise ValueError(f"indices must be provided for {self.split} split")

        # Determine which CIFAR-10 dataset to load based on split
        if self.split in ['train', 'val']:
            # For train/val, load the training dataset
            self.cifar10 = torchvision.datasets.CIFAR10(
                root=self.root, train=True, download=self.download
            )
        else:  # split == 'test'
            # For test, load the test dataset
            self.cifar10 = torchvision.datasets.CIFAR10(
                root=self.root, train=False, download=self.download
            )
        
        # For test split, use None to indicate all data should be used
        if self.split == 'test':
            self.data_indices = None
        else:
            # If max_samples is specified and we have more samples than needed, subsample
            if self.max_samples is not None and self.indices and len(self.indices) > self.max_samples:
                self.data_indices = random.sample(self.indices, self.max_samples)
            else:
                self.data_indices = self.indices
    
    def _get_default_transform(self):
        """Get default transform for CIFAR-10."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def _get_image(self, idx):
        """Get an image by index."""
        # Map to actual CIFAR-10 index
        if self.data_indices is None:
            actual_idx = idx
        else:
            actual_idx = self.data_indices[idx]
        img, _ = self.cifar10[actual_idx]
        return img
    
    def _get_label(self, idx):
        """Get a label by index."""
        # Map to actual CIFAR-10 index
        if self.data_indices is None:
            actual_idx = idx
        else:
            actual_idx = self.data_indices[idx]
        _, label = self.cifar10[actual_idx]
        return label
    
    def __len__(self):
        if self.data_indices is None:
            return len(self.cifar10)
        return len(self.data_indices)


def get_cifar_bag_dataloader(root='./data', train=True, bag_size=5, batch_size=2, 
                            num_workers=4, shuffle=True, indices=None):
    """Get dataloader for bag-level training."""
    dataset = CIFAR10BagDataset(root=root, train=train, bag_size=bag_size, 
                                indices=indices)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_cifar_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                      num_workers=4, shuffle=False, indices=None, max_samples=None):
    """Get dataloader for single image evaluation."""
    dataset = CIFAR10SingleImageDataset(root=root, split=split, indices=indices, 
                                         max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader