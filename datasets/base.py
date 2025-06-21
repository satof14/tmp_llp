import torch
from torch.utils.data import Dataset
import numpy as np
import random
from abc import ABC, abstractmethod


class BaseBagDataset(Dataset, ABC):
    """Base class for bag-level datasets with label proportions."""
    
    def __init__(self, bag_size=5, transform=None):
        self.bag_size = bag_size
        self.transform = transform
        self.bags = []
        self.num_classes = None
        self.class_indices = {}
    
    @abstractmethod
    def _load_data(self):
        """Load data and organize by class. Must set self.class_indices and self.num_classes."""
        pass
    
    @abstractmethod
    def _get_default_transform(self, is_train):
        """Get default transform for the dataset."""
        pass
    
    @abstractmethod
    def _get_image(self, idx):
        """Get an image by index."""
        pass
    
    def _create_bags(self):
        """Create bags with balanced label proportions."""
        bags = []
        
        # Create a copy of class_indices to track remaining samples
        remaining_class_indices = {k: list(v) for k, v in self.class_indices.items()}
        
        # Create balanced bags with known proportions
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        num_bags = total_samples // self.bag_size
        
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
                available_classes = [i for i in range(self.num_classes) 
                                   if len(remaining_class_indices[i]) > counts[i]]
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
                continue
            
            # Sample images according to counts
            bag_indices = []
            bag_labels = []
            
            for class_idx, count in enumerate(counts):
                if count > 0 and class_idx in remaining_class_indices:
                    available_indices = remaining_class_indices[class_idx]
                    if len(available_indices) >= count:
                        sampled_indices = random.sample(available_indices, count)
                        bag_indices.extend(sampled_indices)
                        bag_labels.extend([class_idx] * count)
                        
                        # Remove sampled indices from remaining
                        for idx in sampled_indices:
                            remaining_class_indices[class_idx].remove(idx)
            
            # Skip if bag is not full
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
            img = self._get_image(img_idx)
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Stack images
        images = torch.stack(images)  # (bag_size, C, H, W)
        proportions = torch.tensor(bag['proportions'], dtype=torch.float32)
        
        return images, proportions


class BaseSingleImageDataset(Dataset, ABC):
    """Base class for single image datasets."""
    
    def __init__(self, transform=None):
        self.transform = transform
    
    @abstractmethod
    def _load_data(self):
        """Load data. Must set self.data and self.targets."""
        pass
    
    @abstractmethod
    def _get_default_transform(self):
        """Get default transform for the dataset."""
        pass
    
    @abstractmethod
    def _get_image(self, idx):
        """Get an image by index."""
        pass
    
    @abstractmethod
    def _get_label(self, idx):
        """Get a label by index."""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return the size of the dataset."""
        pass
    
    def __getitem__(self, idx):
        img = self._get_image(idx)
        label = self._get_label(idx)
        
        if self.transform:
            img = self.transform(img)
        
        # Add batch dimension to match bag format
        img = img.unsqueeze(0)  # (1, C, H, W)
        return img, label