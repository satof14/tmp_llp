import torch
import numpy as np
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