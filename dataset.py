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

def compute_channel_stats_from_indices(image_paths, indices):
    """
    Calculates per-channel mean and std from images at the given indices.
    
    Args:
        image_paths: List of all image paths
        indices: List of indices to compute stats from (e.g., training indices)
    
    Returns:
        Dictionary with 'mean' and 'std' lists for RGB channels
    """
    import numpy as np
    from PIL import Image

    print("Calculating channel statistics from training indices...")

    if not indices:
        raise ValueError("No indices provided for channel statistics calculation.")

    sums = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for idx in indices:
        img_path = image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        sums += arr.sum(axis=(0,1))
        sums_sq += (arr**2).sum(axis=(0,1))
        total_pixels += h * w

    print(f"[Calculating stats] Number of training images: {len(indices)}")
    print(f"[Calculating stats] Number of pixels: {total_pixels}")
    print(f"[Calculating stats] Sum of pixel values per channel: {sums}")
    print(f"[Calculating stats] Sum of squared pixel values per channel: {sums_sq}")

    mean = sums / total_pixels
    var  = sums_sq / total_pixels - mean**2

    # clip negative variance due to floating point errors
    var  = np.clip(var, 0, None)
    std  = np.sqrt(var)
    # avoid zero std channels to prevent division by zero
    std = np.where(std == 0, 1e-6, std)

    # cast to float32 for compatibility
    return {
        'mean': mean.astype(np.float32).tolist(),
        'std' : std.astype(np.float32).tolist()
    }


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
    
    def __init__(self, root='./data', train=True, bag_size=5, transform=None, download=True, 
                 train_indices=None, val_indices=None):
        self.bag_size = bag_size
        self.transform = transform if transform else self._get_default_transform(train)
        self.train_indices = train_indices
        self.val_indices = val_indices
        
        # Load CIFAR-10 dataset
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download
        )
        
        # Filter indices based on train/val split if provided
        if train and train_indices is not None:
            # Use only training indices for bag creation
            self.available_indices = set(train_indices)
        else:
            # Use all indices (for test or when no split is provided)
            self.available_indices = set(range(len(self.cifar10)))
        
        # Group indices by class (only available indices)
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.cifar10):
            if idx in self.available_indices:
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
        
        # Create a copy of class_indices to track remaining samples
        remaining_class_indices = {k: list(v) for k, v in self.class_indices.items()}
        
        # Create balanced bags with known proportions
        num_bags = len(self.cifar10) // self.bag_size
        
        for _ in range(num_bags):
            # Check if we have enough samples left
            total_remaining = sum(len(indices) for indices in remaining_class_indices.values())
            if total_remaining < self.bag_size:
                break
                
            # Randomly sample distribution of classes in the bag
            proportions = np.random.dirichlet(np.ones(self.num_classes))
            
            # Convert proportions to counts
            counts = np.round(proportions * self.bag_size).astype(int)
            
            # Adjust counts to ensure sum equals bag_size
            diff = self.bag_size - counts.sum()
            if diff > 0:
                # Add to random classes with available samples
                available_classes = [i for i in range(self.num_classes) if len(remaining_class_indices[i]) > counts[i]]
                if len(available_classes) >= diff:
                    indices = np.random.choice(available_classes, diff, replace=False)
                    for idx in indices:
                        counts[idx] += 1
            elif diff < 0:
                # Remove from random classes with count > 0
                nonzero_indices = np.where(counts > 0)[0]
                indices = np.random.choice(nonzero_indices, -diff, replace=False)
                for idx in indices:
                    counts[idx] -= 1
            
            # Check if we have enough samples for each class
            valid_bag = True
            for class_idx, count in enumerate(counts):
                if count > len(remaining_class_indices[class_idx]):
                    valid_bag = False
                    break
            
            if not valid_bag:
                break
            
            # Sample images according to counts
            bag_indices = []
            bag_labels = []
            
            for class_idx, count in enumerate(counts):
                if count > 0:
                    sampled_indices = random.sample(remaining_class_indices[class_idx], count)
                    bag_indices.extend(sampled_indices)
                    bag_labels.extend([class_idx] * count)
                    
                    # Remove sampled indices from remaining
                    for idx in sampled_indices:
                        remaining_class_indices[class_idx].remove(idx)
            
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
    
    def __init__(self, root='./data', train=False, transform=None, download=True, indices=None, max_samples=None):
        self.transform = transform if transform else self._get_default_transform()
        self.indices = indices
        
        # Load full CIFAR-10 dataset
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=self.transform
        )
        
        # Indices must be provided for single image dataset
        if indices is None:
            raise ValueError("indices must be provided for CIFAR10SingleImageDataset")
        
        # If max_samples is specified and we have more samples than needed, subsample first
        if max_samples is not None and len(indices) > max_samples:
            self.data_indices = random.sample(indices, max_samples)
        else:
            self.data_indices = indices
        
    def _get_default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        # Map to actual CIFAR-10 index
        actual_idx = self.data_indices[idx]
        img, label = self.cifar10[actual_idx]
        # Add batch dimension to match bag format
        img = img.unsqueeze(0)  # (1, C, H, W)
        return img, label


def get_bag_dataloader(root='./data', train=True, bag_size=5, batch_size=2, 
                       num_workers=4, shuffle=True, train_indices=None, val_indices=None):
    """Get dataloader for bag-level training."""
    dataset = CIFAR10BagDataset(root=root, train=train, bag_size=bag_size, 
                               train_indices=train_indices, val_indices=val_indices)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_single_image_dataloader(root='./data', train=False, batch_size=100, 
                                num_workers=4, shuffle=False, indices=None, max_samples=None):
    """Get dataloader for single image evaluation."""
    dataset = CIFAR10SingleImageDataset(root=root, train=train, indices=indices, max_samples=max_samples)
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
                 val_split=0.2, random_seed=42, train_indices=None, val_indices=None, channel_stats=None):
        self.bag_size = bag_size
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.channel_stats = channel_stats
        self.transform = transform
        
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
            
            # For train split, use provided train_indices
            if split == 'train':
                if self.train_indices is None:
                    raise ValueError("train_indices must be provided for train split")
                self.data = [all_data[i] for i in self.train_indices]
                self.targets = [all_targets[i] for i in self.train_indices]
            else:  # split == 'val'
                # This should not be used for validation in the new setup
                raise ValueError("MIFCMBagDataset should not be used for validation split")
                
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
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform(is_train=(split == 'train'))
    
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
        if self.channel_stats is None:
            raise ValueError("channel_stats must be provided")
            
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
                 val_split=0.2, random_seed=42, train_indices=None, val_indices=None, channel_stats=None, max_samples=None):
        self.split = split
        self.val_split = val_split
        self.random_seed = random_seed
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.channel_stats = channel_stats
        self.max_samples = max_samples
        self.transform = transform
        
        # Build path to dataset
        dataset_path = os.path.join(root, "dataset_preprocessed_mokushi_screening_3classes_train_test")
        
        self.data = []
        self.targets = []
        
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load data based on split
        if split in ['train', 'val']:
            # Load all data from train folder
            train_path = os.path.join(dataset_path, "train")
            all_data, all_targets = self._load_images_from_path(train_path, label_mapping)
            
            if split == 'train':
                if self.train_indices is None:
                    raise ValueError("train_indices must be provided for train split")
                # Apply max_samples to indices first
                indices_to_use = self.train_indices
                if self.max_samples is not None and len(indices_to_use) > self.max_samples:
                    indices_to_use = random.sample(indices_to_use, self.max_samples)
                self.data = [all_data[i] for i in indices_to_use]
                self.targets = [all_targets[i] for i in indices_to_use]
            else:  # split == 'val'
                if self.val_indices is None:
                    raise ValueError("val_indices must be provided for val split")
                # Apply max_samples to indices first
                indices_to_use = self.val_indices
                if self.max_samples is not None and len(indices_to_use) > self.max_samples:
                    indices_to_use = random.sample(indices_to_use, self.max_samples)
                self.data = [all_data[i] for i in indices_to_use]
                self.targets = [all_targets[i] for i in indices_to_use]
                
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
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform()
        
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
        if self.channel_stats is None:
            raise ValueError("channel_stats must be provided")
            
        mean = self.channel_stats['mean']
        std = self.channel_stats['std']
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
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
                             num_workers=4, shuffle=True, val_split=0.2, random_seed=42,
                             train_indices=None, val_indices=None, channel_stats=None):
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
        train_indices: Pre-computed training indices (optional)
        val_indices: Pre-computed validation indices (optional)
        channel_stats: Channel statistics for normalization (required)
    """
    dataset = MIFCMBagDataset(root=root, split=split, bag_size=bag_size, 
                              val_split=val_split, random_seed=random_seed,
                              train_indices=train_indices, val_indices=val_indices,
                              channel_stats=channel_stats)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_mifcm_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                      num_workers=4, shuffle=False, val_split=0.2, random_seed=42,
                                      train_indices=None, val_indices=None, channel_stats=None, max_samples=None):
    """Get dataloader for MIFCM single image evaluation.
    
    Args:
        root: Root directory of the dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size for dataloader
        num_workers: Number of workers for dataloader
        shuffle: Whether to shuffle the data
        val_split: Fraction of train data to use for validation (only used when split='train' or 'val')
        random_seed: Random seed for reproducible train/val split
        train_indices: Pre-computed training indices (optional)
        val_indices: Pre-computed validation indices (optional)
        channel_stats: Channel statistics for normalization (required)
        max_samples: Maximum number of samples to use (optional)
    """
    dataset = MIFCMSingleImageDataset(root=root, split=split, 
                                      val_split=val_split, random_seed=random_seed,
                                      train_indices=train_indices, val_indices=val_indices,
                                      channel_stats=channel_stats, max_samples=max_samples)
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
    
    def __init__(self, root='./data', split='train', bag_size=5, transform=None, channel_stats=None):
        self.bag_size = bag_size
        self.channel_stats = channel_stats
        # Initially set transform to None for dynamic calculation
        self.transform = transform
        
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
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform(is_train=(split == 'train'))
    
    def get_training_bags_indices(self):
        """Get the indices used in training bags for channel stats calculation."""
        train_bags = []
        for bag in self.bags:
            train_bags.append(bag['indices'])
        return train_bags
        
    def _get_default_transform(self, is_train):
        if self.channel_stats is None:
            raise ValueError("channel_stats must be provided")
            
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
    
    def __init__(self, root='./data', split='test', transform=None, channel_stats=None, max_samples=None):
        self.channel_stats = channel_stats
        self.max_samples = max_samples
        # Initially set transform to None for dynamic calculation
        self.transform = transform
        
        self.data = []
        self.targets = []
        
        # Load data from text files
        list_file = f'{split}.txt'
        with open(os.path.join(root, list_file), 'r') as f:
            lines = f.readlines()
        
        # Apply max_samples before processing if specified
        if self.max_samples is not None and len(lines) > self.max_samples:
            lines = random.sample(lines, self.max_samples)
        
        # Process selected lines
        for line in lines:
            rel_path, label_idx = line.strip().split()
            img_path = os.path.join(root, split, rel_path)
            self.data.append(img_path)
            self.targets.append(int(label_idx))
        
        # Set transform if not provided
        if self.transform is None:
            self.transform = self._get_default_transform()
        
    def _get_default_transform(self):
        if self.channel_stats is None:
            raise ValueError("channel_stats must be provided")
            
        mean = self.channel_stats['mean']
        std = self.channel_stats['std']
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
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
                                           num_workers=4, shuffle=True, channel_stats=None):
    """Get dataloader for Human Somatic Small bag-level training."""
    dataset = HumanSomaticSmallBagDataset(root=root, split=split, bag_size=bag_size, channel_stats=channel_stats)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def get_human_somatic_small_single_image_dataloader(root='./data', split='test', batch_size=100, 
                                                    num_workers=4, shuffle=False, channel_stats=None, max_samples=None):
    """Get dataloader for Human Somatic Small single image evaluation."""
    dataset = HumanSomaticSmallSingleImageDataset(root=root, split=split, channel_stats=channel_stats, max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader