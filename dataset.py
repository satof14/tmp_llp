import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
import random
import os
from PIL import Image
from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Unified dataset splitter for train/validation splits."""
    
    @staticmethod
    def split_indices(dataset_size, valid_ratio=0.1, seed=42, stratify=None):
        """Split dataset indices into train/validation sets.
        
        Args:
            dataset_size: Size of the dataset
            valid_ratio: Ratio of validation data (default: 0.1)
            seed: Random seed for reproducibility
            stratify: Labels for stratified splitting (optional)
        
        Returns:
            train_indices, val_indices
        """
        indices = list(range(dataset_size))
        
        if stratify is not None:
            # Stratified split
            train_idx, val_idx = train_test_split(
                indices, test_size=valid_ratio, random_state=seed, stratify=stratify
            )
        else:
            # Random split
            torch.manual_seed(seed)
            val_size = int(valid_ratio * dataset_size)
            train_size = dataset_size - val_size
            train_idx, val_idx = torch.randperm(dataset_size).split([train_size, val_size])
            train_idx, val_idx = train_idx.tolist(), val_idx.tolist()
        
        return train_idx, val_idx
    
    @staticmethod
    def verify_no_overlap(train_indices, val_indices):
        """Quick overlap check."""
        overlap = set(train_indices) & set(val_indices)
        if overlap:
            raise ValueError(f"Found {len(overlap)} overlapping indices!")
        else:
            print(f"No overlap found between train ({len(train_indices)}) and validation ({len(val_indices)}) indices")
        return True


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
    """MIFCM 3-classes dataset for bag-level training with label proportions.
    
    This class supports train/validation split from the original train set.
    """
    
    def __init__(self, root='./data', split='train', bag_size=5, transform=None, 
                 val_split=0.2, random_seed=42):
        self.bag_size = bag_size
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed
        self.transform = transform if transform else self._get_default_transform(split == 'train')
        
        # Build path to dataset
        dataset_path = os.path.join(root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data based on split
        if split in ['train', 'val']:
            # Load all data from train folder
            train_path = os.path.join(dataset_path, "train")
            all_data, all_targets = self._load_images_from_path(train_path, label_mapping)
            
            # Split using unified splitter
            train_idx, val_idx = DatasetSplitter.split_indices(
                len(all_data), val_split, random_seed, stratify=all_targets
            )
            DatasetSplitter.verify_no_overlap(train_idx, val_idx)
            
            if split == 'train':
                self.data = [all_data[i] for i in train_idx]
                self.targets = [all_targets[i] for i in train_idx]
            else:  # split == 'val'
                self.data = [all_data[i] for i in val_idx]
                self.targets = [all_targets[i] for i in val_idx]
                
        else:  # split == 'test'
            # Load from test folder
            test_path = os.path.join(dataset_path, "test")
            for label_name in os.listdir(test_path):
                label_path = os.path.join(test_path, label_name)
                if os.path.isdir(label_path) and label_name in label_mapping:
                    label_idx = label_mapping[label_name]
                    for file in os.listdir(label_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                            img_path = os.path.join(label_path, file)
                            self.data.append(img_path)
                            self.targets.append(label_idx)
        
        # Group by class
        for idx, target in enumerate(self.targets):
            self.class_indices[target].append(idx)
        
        self.num_classes = 3
        self.class_indices = {k: list(v) for k, v in self.class_indices.items()}
        
        # Create bags
        self.bags = self._create_bags()
    
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
        if is_train:
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
    """MIFCM 3-classes dataset for single image evaluation.
    
    This class supports train/validation split from the original train set.
    """
    
    def __init__(self, root='./data', split='test', transform=None, 
                 val_split=0.2, random_seed=42):
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed
        self.transform = transform if transform else self._get_default_transform()
        
        # Build path to dataset
        dataset_path = os.path.join(root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data based on split
        if split in ['train', 'val']:
            # Load from train folder and split
            train_path = os.path.join(dataset_path, "train")
            all_data = []
            all_targets = []
            
            for label_name in os.listdir(train_path):
                label_path = os.path.join(train_path, label_name)
                if os.path.isdir(label_path) and label_name in label_mapping:
                    label_idx = label_mapping[label_name]
                    for file in os.listdir(label_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                            img_path = os.path.join(label_path, file)
                            all_data.append(img_path)
                            all_targets.append(label_idx)
            
            # Split data into train and validation
            train_data, val_data, train_targets, val_targets = self._split_data(
                all_data, all_targets, val_split, random_seed
            )
            
            if split == 'train':
                self.data = train_data
                self.targets = train_targets
            else:  # split == 'val'
                self.data = val_data
                self.targets = val_targets
                
        else:  # split == 'test'
            # Load from test folder
            test_path = os.path.join(dataset_path, "test")
            for label_name in os.listdir(test_path):
                label_path = os.path.join(test_path, label_name)
                if os.path.isdir(label_path) and label_name in label_mapping:
                    label_idx = label_mapping[label_name]
                    for file in os.listdir(label_path):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                            img_path = os.path.join(label_path, file)
                            self.data.append(img_path)
                            self.targets.append(label_idx)
        
    def _split_data(self, data, targets, val_split, random_seed):
        """Split data into train and validation sets while preserving class distribution."""
        train_data, val_data, train_targets, val_targets = train_test_split(
            data, targets, test_size=val_split, random_state=random_seed,
            stratify=targets  # Preserve class distribution
        )
        
        return train_data, val_data, train_targets, val_targets
    
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


def get_mifcm_bag_dataloader(root='./data', split='train', bag_size=5, batch_size=2, 
                             num_workers=4, shuffle=True, val_split=0.2, random_seed=42):
    """Get dataloader for MIFCM bag-level training.
    
    Args:
        root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        bag_size: Number of images per bag
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle the data
        val_split: Fraction of train data to use for validation (only used when split='train' or 'val')
        random_seed: Random seed for reproducible train/val split
    """
    dataset = MIFCMBagDataset(root=root, split=split, bag_size=bag_size, 
                              val_split=val_split, random_seed=random_seed)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_mifcm_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                      num_workers=4, shuffle=False, val_split=0.2, random_seed=42):
    """Get dataloader for MIFCM single image evaluation.
    
    Args:
        root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle the data
        val_split: Fraction of train data to use for validation (only used when split='train' or 'val')
        random_seed: Random seed for reproducible train/val split
    """
    dataset = MIFCMSingleImageDataset(root=root, split=split, 
                                      val_split=val_split, random_seed=random_seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


class HumanSomaticSmallBagDataset(Dataset):
    """Human Somatic Small dataset for bag-level training with label proportions."""
    
    def __init__(self, root='./data', split='train', bag_size=5, transform=None):
        self.bag_size = bag_size
        self.transform = transform if transform else self._get_default_transform(split == 'train')
        
        # Build path to dataset - same structure as llp_vat
        self.root = root
        
        # Group indices by class using text files
        self.class_indices = defaultdict(list)
        self.data = []
        self.targets = []
        
        # Load data from text files (train.txt, val.txt, test.txt)
        list_file = f'{split}.txt'
        with open(os.path.join(root, list_file), 'r') as f:
            for line in f:
                rel_path, label_idx = line.strip().split()
                img_path = os.path.join(root, split, rel_path)
                idx = len(self.data)
                self.data.append(img_path)
                self.targets.append(int(label_idx))
                self.class_indices[int(label_idx)].append(idx)
        
        self.num_classes = 3
        self.class_indices = {k: list(v) for k, v in self.class_indices.items()}
        
        # Create bags
        self.bags = self._create_bags()
        
    def _get_default_transform(self, train):
        if train:
            return transforms.Compose([
                transforms.RandomCrop(128, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            return transforms.Compose([
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


class HumanSomaticSmallSingleImageDataset(Dataset):
    """Human Somatic Small dataset for single image evaluation."""
    
    def __init__(self, root='./data', split='test', transform=None):
        self.transform = transform if transform else self._get_default_transform()
        
        self.data = []
        self.targets = []
        
        # Load data from text files
        list_file = f'{split}.txt'
        with open(os.path.join(root, list_file), 'r') as f:
            for line in f:
                rel_path, label_idx = line.strip().split()
                img_path = os.path.join(root, split, rel_path)
                self.data.append(img_path)
                self.targets.append(int(label_idx))
        
    def _get_default_transform(self):
        return transforms.Compose([
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


def get_human_somatic_small_bag_dataloader(root='./data', split='train', bag_size=5, batch_size=2, 
                                           num_workers=4, shuffle=True):
    """Get dataloader for Human Somatic Small bag-level training."""
    dataset = HumanSomaticSmallBagDataset(root=root, split=split, bag_size=bag_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_human_somatic_small_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                                    num_workers=4, shuffle=False):
    """Get dataloader for Human Somatic Small single image evaluation."""
    dataset = HumanSomaticSmallSingleImageDataset(root=root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader