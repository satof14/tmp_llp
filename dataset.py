import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
import random
import os
from PIL import Image


class CIFAR10BagDataset(Dataset):
    """CIFAR-10 dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', train=True, bag_size=5, transform=None, download=True):
        self.bag_size = bag_size
        self.transform = transform if transform else self._get_default_transform(train)
        
        # Load CIFAR-10 dataset
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download
        )
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.cifar10):
            self.class_indices[label].append(idx)
            
        # Convert to lists for easier access
        self.class_indices = {k: list(v) for k, v in self.class_indices.items()}
        self.num_classes = len(self.class_indices)
        
        # Create bags
        self.bags = self._create_bags()
        
    def _get_default_transform(self, train):
        if train:
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

    def _create_bags(self):
        bags = []
        
        # Create balanced bags with known proportions
        num_bags = len(self.cifar10) // self.bag_size
        
        for _ in range(num_bags):
            # Randomly sample distribution of classes in the bag
            proportions = np.random.dirichlet(np.ones(self.num_classes))
            
            # Convert proportions to counts
            counts = np.round(proportions * self.bag_size).astype(int)
            
            # Adjust counts to ensure sum equals bag_size
            diff = self.bag_size - counts.sum()
            if diff > 0:
                # Add to random classes
                indices = np.random.choice(self.num_classes, diff, replace=False)
                for idx in indices:
                    counts[idx] += 1
            elif diff < 0:
                # Remove from random classes with count > 0
                nonzero_indices = np.where(counts > 0)[0]
                indices = np.random.choice(nonzero_indices, -diff, replace=False)
                for idx in indices:
                    counts[idx] -= 1
            
            # Sample images according to counts
            bag_indices = []
            bag_labels = []
            
            for class_idx, count in enumerate(counts):
                if count > 0:
                    sampled_indices = random.sample(self.class_indices[class_idx], count)
                    bag_indices.extend(sampled_indices)
                    bag_labels.extend([class_idx] * count)
            
            # Calculate actual proportions
            actual_proportions = np.zeros(self.num_classes)
            for label in bag_labels:
                actual_proportions[label] += 1
            actual_proportions /= self.bag_size
            
            bags.append({
                'indices': bag_indices,
                'labels': bag_labels,
                'proportions': actual_proportions
            })
        
        return bags
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        images = []
        
        for img_idx in bag['indices']:
            img, _ = self.cifar10[img_idx]
            img = self.transform(img)
            images.append(img)
        
        # Stack images
        images = torch.stack(images)  # (bag_size, C, H, W)
        proportions = torch.tensor(bag['proportions'], dtype=torch.float32)
        
        return images, proportions


class CIFAR10SingleImageDataset(Dataset):
    """CIFAR-10 dataset for single image evaluation."""
    
    def __init__(self, root='./data', train=False, transform=None, download=True):
        self.transform = transform if transform else self._get_default_transform()
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=self.transform
        )
        
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        img, label = self.cifar10[idx]
        # Add batch dimension to match bag format
        img = img.unsqueeze(0)  # (1, C, H, W)
        return img, label


def get_bag_dataloader(root='./data', train=True, bag_size=5, batch_size=2, 
                       num_workers=4, shuffle=True):
    """Get dataloader for bag-level training."""
    dataset = CIFAR10BagDataset(root=root, train=train, bag_size=bag_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_single_image_dataloader(root='./data', train=False, batch_size=100, 
                                num_workers=4, shuffle=False):
    """Get dataloader for single image evaluation."""
    dataset = CIFAR10SingleImageDataset(root=root, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class MIFCMBagDataset(Dataset):
    """MIFCM 3-classes dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', train=True, bag_size=5, transform=None):
        self.bag_size = bag_size
        self.transform = transform if transform else self._get_default_transform(train)
        
        # Build path to dataset
        dataset_path = os.path.join(root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        split_path = os.path.join(dataset_path, "train" if train else "test")
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data and group by class
        for label_name in os.listdir(split_path):
            label_path = os.path.join(split_path, label_name)
            if os.path.isdir(label_path) and label_name in label_mapping:
                label_idx = label_mapping[label_name]
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        img_path = os.path.join(label_path, file)
                        idx = len(self.data)
                        self.data.append(img_path)
                        self.targets.append(label_idx)
                        self.class_indices[label_idx].append(idx)
        
        self.num_classes = 3
        self.class_indices = {k: list(v) for k, v in self.class_indices.items()}
        
        # Create bags
        self.bags = self._create_bags()
        
    def _get_default_transform(self, train):
        if train:
            return transforms.Compose([
                transforms.Resize(64),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            return transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

    def _create_bags(self):
        bags = []
        
        # Create balanced bags with known proportions
        num_bags = len(self.data) // self.bag_size
        
        for _ in range(num_bags):
            # Randomly sample distribution of classes in the bag
            proportions = np.random.dirichlet(np.ones(self.num_classes))
            
            # Convert proportions to counts
            counts = np.round(proportions * self.bag_size).astype(int)
            
            # Adjust counts to ensure sum equals bag_size
            diff = self.bag_size - counts.sum()
            if diff > 0:
                # Add to random classes
                indices = np.random.choice(self.num_classes, diff, replace=False)
                for idx in indices:
                    counts[idx] += 1
            elif diff < 0:
                # Remove from random classes with count > 0
                nonzero_indices = np.where(counts > 0)[0]
                indices = np.random.choice(nonzero_indices, -diff, replace=False)
                for idx in indices:
                    counts[idx] -= 1
            
            # Sample images according to counts
            bag_indices = []
            bag_labels = []
            
            for class_idx, count in enumerate(counts):
                if count > 0 and class_idx in self.class_indices:
                    available_indices = self.class_indices[class_idx]
                    if len(available_indices) >= count:
                        sampled_indices = random.sample(available_indices, count)
                        bag_indices.extend(sampled_indices)
                        bag_labels.extend([class_idx] * count)
            
            # Skip if bag is empty or too small
            if len(bag_indices) < self.bag_size:
                continue
                
            # Calculate actual proportions
            actual_proportions = np.zeros(self.num_classes)
            for label in bag_labels:
                actual_proportions[label] += 1
            actual_proportions /= len(bag_labels)
            
            bags.append({
                'indices': bag_indices,
                'labels': bag_labels,
                'proportions': actual_proportions
            })
        
        return bags
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        images = []
        
        for img_idx in bag['indices']:
            img_path = self.data[img_idx]
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            images.append(img)
        
        # Stack images
        images = torch.stack(images)  # (bag_size, C, H, W)
        proportions = torch.tensor(bag['proportions'], dtype=torch.float32)
        
        return images, proportions


class MIFCMSingleImageDataset(Dataset):
    """MIFCM 3-classes dataset for single image evaluation."""
    
    def __init__(self, root='./data', train=False, transform=None):
        self.transform = transform if transform else self._get_default_transform()
        
        # Build path to dataset
        dataset_path = os.path.join(root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        split_path = os.path.join(dataset_path, "train" if train else "test")
        
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data
        for label_name in os.listdir(split_path):
            label_path = os.path.join(split_path, label_name)
            if os.path.isdir(label_path) and label_name in label_mapping:
                label_idx = label_mapping[label_name]
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        img_path = os.path.join(label_path, file)
                        self.data.append(img_path)
                        self.targets.append(label_idx)
        
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        
        # Add batch dimension to match bag format
        img = img.unsqueeze(0)  # (1, C, H, W)
        return img, label


def get_mifcm_bag_dataloader(root='./data', train=True, bag_size=5, batch_size=2, 
                             num_workers=4, shuffle=True):
    """Get dataloader for MIFCM bag-level training."""
    dataset = MIFCMBagDataset(root=root, train=train, bag_size=bag_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_mifcm_single_image_dataloader(root='./data', train=False, batch_size=100, 
                                      num_workers=4, shuffle=False):
    """Get dataloader for MIFCM single image evaluation."""
    dataset = MIFCMSingleImageDataset(root=root, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader