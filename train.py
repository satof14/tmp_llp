import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import numpy as np
import time

from model import LLPAttentionModel
from dataset import get_bag_dataloader, get_single_image_dataloader, get_mifcm_bag_dataloader, get_mifcm_single_image_dataloader, get_human_somatic_small_bag_dataloader, get_human_somatic_small_single_image_dataloader, DatasetSplitter, compute_channel_stats_from_bags
from collections import Counter


def build_optimizer(model, config):
    """Build optimizer based on configuration."""
    optimizer_type = config['optimizer'].lower()
    lr = config['learning_rate']
    weight_decay = config['weight_decay']
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)),
            eps=config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"Using {optimizer_type.upper()} optimizer with lr={lr}")
    return optimizer


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameters_string(model):
    """Get a string representation of model parameters."""
    trainable_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    return f"Trainable parameters: {trainable_params:,} | Total parameters: {total_params:,}"


def get_class_distribution(dataset, dataset_name):
    """Get class distribution for a dataset."""
    if hasattr(dataset, 'targets'):
        # Single image dataset
        labels = dataset.targets
    elif hasattr(dataset, 'cifar10'):
        # CIFAR10 wrapper
        labels = [dataset.cifar10[i][1] for i in range(len(dataset.cifar10))]
    elif hasattr(dataset, 'bags'):
        # Bag dataset - get all individual labels
        labels = []
        for bag in dataset.bags:
            labels.extend(bag['labels'])
    else:
        return None
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"Class distribution ({dataset_name}):")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_idx}: {count:,} samples ({percentage:.1f}%)")
    print(f"  Total samples: {total_samples:,}")
    
    return class_counts


def get_single_image_dataset_distribution(loader, dataset_name):
    """Get class distribution for single image datasets (validation/test)."""
    dataset = loader.dataset
    
    # Handle different dataset types
    if hasattr(dataset, 'dataset'):
        # This is a Subset wrapping the original dataset
        original_dataset = dataset.dataset
        if hasattr(original_dataset, 'targets'):
            # Get targets for the subset indices
            subset_targets = [original_dataset.targets[i] for i in dataset.indices]
            labels = subset_targets
        else:
            # Fallback: iterate through subset
            labels = []
            for i in dataset.indices:
                _, label = original_dataset[i]
                labels.append(label)
    elif hasattr(dataset, 'targets'):
        # Direct dataset with targets attribute
        labels = dataset.targets
    else:
        # Fallback: iterate through all samples
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print(f"Class distribution ({dataset_name}):")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_idx}: {count:,} samples ({percentage:.1f}%)")
    print(f"  Total {dataset_name.lower()} samples: {total_samples:,}")
    
    return class_counts


def get_train_bags_class_distribution(train_loader):
    """Get class distribution specifically from training bags only."""
    # Get the actual training bags (after split)
    if hasattr(train_loader.dataset, 'dataset'):
        # This is a Subset wrapping the original bag dataset
        full_bag_dataset = train_loader.dataset.dataset
        train_indices = train_loader.dataset.indices
        
        # Get labels only from training bags
        train_labels = []
        for bag_idx in train_indices:
            bag = full_bag_dataset.bags[bag_idx]
            train_labels.extend(bag['labels'])
    else:
        # Direct bag dataset (no split)
        bag_dataset = train_loader.dataset
        train_labels = []
        for bag in bag_dataset.bags:
            train_labels.extend(bag['labels'])
    
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    
    print(f"Class distribution (Train):")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total_samples) * 100
        print(f"  Class {class_idx}: {count:,} samples ({percentage:.1f}%)")
    print(f"  Total training samples in bags: {total_samples:,}")
    
    return class_counts


def print_dataset_info(train_loader, val_loader, test_loader, config):
    """Print comprehensive dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Basic info
    print(f"Dataset: {config.get('dataset', 'CIFAR-10')}")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Bag size: {config['bag_size']}")
    print(f"Mini-batch size: {config['mini_batch_size']}")
    
    # Dataset sizes
    print(f"\nDataset splits:")
    print(f"  Train bags: {len(train_loader.dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    
    # Class distributions
    print(f"\nClass distributions:")
    
    # Training set class distribution (from training bags only)
    get_train_bags_class_distribution(train_loader)
    
    # Validation set class distribution
    get_single_image_dataset_distribution(val_loader, "Validation")
    
    # Test set class distribution
    get_single_image_dataset_distribution(test_loader, "Test")
    
    print("="*60)


def print_model_info(model, optimizer, config, log_dir=None):
    """Print comprehensive model information and save to file."""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    # Model architecture info
    print(f"Model: LLPAttentionModel")
    print(f"Image size: {config.get('img_size', 32)}x{config.get('img_size', 32)}")
    print(f"Patch size: {config['patch_size']}x{config['patch_size']}")
    print(f"Embedding dimension: {config['embed_dim']}")
    print(f"Number of attention heads: {config['num_heads']}")
    print(f"Number of transformer layers: {config['L']}")
    print(f"MLP ratio: {config.get('mlp_ratio', 4.0)}")
    print(f"Dropout rate: {config['dropout']}")
    
    # Parameter count
    param_info = parameters_string(model)
    print(f"\n{param_info}")
    
    # Optimizer info
    print(f"\nOptimizer: {optimizer.__class__.__name__}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    if hasattr(optimizer, 'betas'):
        print(f"Betas: {optimizer.param_groups[0]['betas']}")
    if hasattr(optimizer, 'momentum'):
        print(f"Momentum: {optimizer.param_groups[0]['momentum']}")
    
    print("="*60)
    
    # Save model structure to file
    if log_dir:
        model_info_path = os.path.join(log_dir, 'model_structure.txt')
        with open(model_info_path, 'w') as f:
            f.write("MODEL STRUCTURE\n")
            f.write("="*50 + "\n\n")
            f.write(str(model) + "\n\n")
            f.write("MODEL PARAMETERS\n")
            f.write("="*50 + "\n")
            f.write(f"{param_info}\n\n")
            
            # Detailed parameter breakdown
            f.write("PARAMETER BREAKDOWN:\n")
            f.write("-"*30 + "\n")
            total_params = 0
            for name, param in model.named_parameters():
                param_count = param.numel()
                total_params += param_count
                f.write(f"{name:.<40} {param_count:>10,}\n")
            f.write("-"*50 + "\n")
            f.write(f"{'Total':<40} {total_params:>10,}\n")
        
        print(f"Model structure saved to: {model_info_path}")


def format_elapsed_time(seconds):
    """Format elapsed time in days, hours, minutes, seconds."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None, bag_size=1, grad_clip=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, proportions) in enumerate(pbar):
        images = images.to(device)
        proportions = proportions.to(device)
        
        # Forward pass
        logits = model(images)
        log_predictions = torch.log_softmax(logits, dim=1)
        
        # Proportion loss (KLDivLoss expects log probabilities as input)
        loss = criterion(log_predictions, proportions)
        
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on single images."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy




def train(config, log_dir=None):
    """Main training function."""
    # Record start time
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model
    if config.get('dataset') == 'mifcm_3classes_newgate':
        img_size = 64
    elif config.get('dataset') == 'human_somatic_small':
        img_size = 128
    else:
        img_size = 32
    
    # Store img_size in config for model info display
    config['img_size'] = img_size
    
    model = LLPAttentionModel(
        img_size=img_size,
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['L'],
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config['dropout']
    ).to(device)
    
    # Create dataloaders with train/valid split
    if config.get('dataset') == 'mifcm_3classes_newgate':
        # First, load all training data to get indices for train/valid split
        from dataset import MIFCMSingleImageDataset
        import os
        
        # Load all training data paths to determine train/valid split indices
        dataset_path = os.path.join(config['data_root'], "dataset_preprocessed_mokushi_screening_3classes_train_test")
        train_path = os.path.join(dataset_path, "train")
        
        all_data = []
        all_targets = []
        label_mapping = {'G1': 0, 'S': 1, 'G2': 2}
        
        # Load all training data
        for label_name in os.listdir(train_path):
            label_path = os.path.join(train_path, label_name)
            if os.path.isdir(label_path) and label_name in label_mapping:
                label_idx = label_mapping[label_name]
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        img_path = os.path.join(label_path, file)
                        all_data.append(img_path)
                        all_targets.append(label_idx)
        
        # Split indices for train/validation using unified splitter
        train_indices, val_indices = DatasetSplitter.split_indices(
            len(all_data), 
            config.get('valid_ratio', 0.1), 
            config.get('seed', 42),
            stratify=all_targets
        )
        DatasetSplitter.verify_no_overlap(train_indices, val_indices)
        
        print(f"Total training images: {len(all_data)}")
        print(f"Train split: {len(train_indices)} images")
        print(f"Valid split: {len(val_indices)} images")
        
        # Create train and validation bag datasets using pre-split indices
        train_bag_dataset = get_mifcm_bag_dataloader(
            root=config['data_root'],
            split='train',
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True,
            train_indices=train_indices,
            val_indices=val_indices
        ).dataset
        
        val_bag_dataset = get_mifcm_bag_dataloader(
            root=config['data_root'],
            split='val',
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True,
            train_indices=train_indices,
            val_indices=val_indices
        ).dataset
        
        # Calculate channel statistics from training bags only
        train_bags_indices = train_bag_dataset.get_training_bags_indices()
        channel_stats = compute_channel_stats_from_bags(train_bag_dataset, train_bags_indices)
        print("Channel stats:", channel_stats)
        
        # Create transforms with calculated statistics
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        
        # Override dataset transforms
        train_bag_dataset.transform = transform_train
        val_bag_dataset.transform = transform_train
        
        # Create train loader from training bags
        train_loader = DataLoader(
            train_bag_dataset,
            batch_size=config['mini_batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Create validation single-image loader
        val_single_dataset = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            split='val',
            batch_size=100,
            shuffle=False,
            train_indices=train_indices,
            val_indices=val_indices
        ).dataset
        
        # Override single image dataset transform
        val_single_dataset.transform = transform_test
        
        val_loader = DataLoader(
            val_single_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create test loader for final evaluation
        test_loader = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            split='test',
            batch_size=100,
            shuffle=False
        )
        
        # Override test dataset transform
        test_loader.dataset.transform = transform_test
        
        # Create train instance-level dataloader for train accuracy evaluation
        train_instance_loader = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            split='train',
            batch_size=100,
            shuffle=False,
            train_indices=train_indices,
            val_indices=val_indices
        )
        
        # Override train instance dataset transform
        train_instance_loader.dataset.transform = transform_test
    elif config.get('dataset') == 'human_somatic_small':
        # For human_somatic_small, use pre-split validation as in llp_vat
        train_loader = get_human_somatic_small_bag_dataloader(
            root=config['data_root'],
            split='train',
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True
        )
        
        # Calculate channel statistics from training bags only
        train_bags_indices = train_loader.dataset.get_training_bags_indices()
        channel_stats = compute_channel_stats_from_bags(train_loader.dataset, train_bags_indices)
        print("Channel stats:", channel_stats)
        
        # Create transforms with calculated statistics
        transform_train = transforms.Compose([
            transforms.RandomCrop(128, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        
        # Override dataset transforms
        train_loader.dataset.transform = transform_train
        
        val_loader = get_human_somatic_small_single_image_dataloader(
            root=config['data_root'],
            split='val',
            batch_size=100,
            shuffle=False
        )
        
        # Override validation dataset transform
        val_loader.dataset.transform = transform_test
        
        # Create test loader for final evaluation
        test_loader = get_human_somatic_small_single_image_dataloader(
            root=config['data_root'],
            split='test',
            batch_size=100,
            shuffle=False
        )
        
        # Override test dataset transform
        test_loader.dataset.transform = transform_test
        
        # Create train instance-level dataloader for train accuracy evaluation
        train_instance_loader = get_human_somatic_small_single_image_dataloader(
            root=config['data_root'],
            split='train',
            batch_size=100,
            shuffle=False
        )
        
        # Override train instance dataset transform
        train_instance_loader.dataset.transform = transform_test
    else:
        # Create full bag dataset and split for training/validation
        full_bag_dataset = get_bag_dataloader(
            root=config['data_root'],
            train=True,
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True
        ).dataset
        
        # Split bags for training and validation using unified splitter
        train_idx, val_idx = DatasetSplitter.split_indices(
            len(full_bag_dataset), 
            config.get('valid_ratio', 0.1), 
            config.get('seed', 42)
        )
        DatasetSplitter.verify_no_overlap(train_idx, val_idx)
        
        train_bags = Subset(full_bag_dataset, train_idx)
        valid_bags = Subset(full_bag_dataset, val_idx)
        
        # Create train loader from training bags
        train_loader = DataLoader(
            train_bags,
            batch_size=config['mini_batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # For validation, create single-image loader from validation bags
        val_single_images = []
        val_single_labels = []
        for bag_idx in valid_bags.indices:
            bag = full_bag_dataset.bags[bag_idx]
            val_single_images.extend(bag['indices'])
            val_single_labels.extend(bag['labels'])
        
        # Collect training indices for overlap check
        train_single_images_for_check = []
        for bag_idx in train_bags.indices:
            bag = full_bag_dataset.bags[bag_idx]
            train_single_images_for_check.extend(bag['indices'])
        
        # Check for overlap between train and validation sets
        DatasetSplitter.verify_no_overlap(train_single_images_for_check, val_single_images)
        
        # Create single image dataset and subset it
        cifar_single_dataset = get_single_image_dataloader(
            root=config['data_root'],
            train=True,
            batch_size=100,
            shuffle=False
        ).dataset
        
        val_single_dataset = Subset(cifar_single_dataset, val_single_images)
        
        val_loader = DataLoader(
            val_single_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create test loader for final evaluation
        test_loader = get_single_image_dataloader(
            root=config['data_root'],
            train=False,
            batch_size=100,
            shuffle=False
        )
        
        # Create train instance-level dataloader for train accuracy evaluation
        train_instance_loader = get_single_image_dataloader(
            root=config['data_root'],
            train=True,
            batch_size=100,
            shuffle=False
        )
    
    # Print comprehensive dataset information
    print_dataset_info(train_loader, val_loader, test_loader, config)
    
    # Create subset of training data to match validation set size for fair comparison
    val_size = len(val_loader.dataset)
    if len(train_instance_loader.dataset) > val_size:
        # Use train bags to create a subset that matches validation size
        if config.get('dataset') == 'human_somatic_small' or config.get('dataset') == 'mifcm_3classes_newgate':
            # For human_somatic_small and mifcm_3classes_newgate, use random subset
            indices = torch.randperm(len(train_instance_loader.dataset))[:val_size]
            subset_dataset = Subset(train_instance_loader.dataset, indices)
        else:
            # For other datasets, use images from training bags
            train_single_images = []
            for bag_idx in train_bags.indices:
                bag = full_bag_dataset.bags[bag_idx]
                train_single_images.extend(bag['indices'])
            
            # Limit to validation size
            if len(train_single_images) > val_size:
                train_single_images = train_single_images[:val_size]
            
            subset_dataset = Subset(train_instance_loader.dataset, train_single_images)
        
        train_instance_loader = DataLoader(
            subset_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # Create optimizer and loss function
    optimizer = build_optimizer(model, config)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    # Print comprehensive model information
    print_model_info(model, optimizer, config, log_dir)
    
    # Create scheduler with warmup
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        # Create warmup scheduler
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        # Create main scheduler
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=1e-6
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6
        )
    
    # Create tensorboard writer
    tensorboard_dir = os.path.join(log_dir, 'tensorboard') if log_dir else 'results/llp_attention'
    writer = SummaryWriter(tensorboard_dir)
    
    # Training loop
    best_accuracy = 0
    best_epoch = 0
    best_train_instance_accuracy = 0
    best_train_instance_epoch = 0
    best_test_accuracy = 0
    best_test_epoch = 0
    
    for epoch in range(config['epochs']):
        # Train
        avg_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, config['bag_size'], config['grad_clip'])
        current_lr = scheduler.get_last_lr()[0]
        elapsed_time = time.time() - start_time
        
        # Print training info
        print(f'Epoch {epoch}/{config["epochs"]-1}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}, Elapsed: {format_elapsed_time(elapsed_time)}')
        
        # Evaluate
        if (epoch + 1) % config['eval_interval'] == 0:
            # Evaluate on validation set
            accuracy = evaluate(model, val_loader, device)
            
            # Evaluate on test set
            test_accuracy = evaluate(model, test_loader, device)
            
            # Evaluate on training set (instance-level)
            train_instance_accuracy = evaluate(model, train_instance_loader, device)
            
            # Track best accuracies
            if train_instance_accuracy > best_train_instance_accuracy:
                best_train_instance_accuracy = train_instance_accuracy
                best_train_instance_epoch = epoch
            
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_epoch = epoch
            
            # Save best model based on validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': accuracy,
                    'test_accuracy': test_accuracy,
                    'config': config
                }, os.path.join(log_dir, 'best_model.pth') if log_dir else 'best_model.pth')
                print(f'Saved best model with val accuracy: {accuracy:.4f}, test accuracy: {test_accuracy:.4f}')
            
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch}/{config["epochs"]-1} | Elapsed: {format_elapsed_time(elapsed_time)}')
            print(f'Train Instance Accuracy: {train_instance_accuracy:.4f} (Best: {best_train_instance_accuracy:.4f} @ Epoch {best_train_instance_epoch})')
            print(f'           Val Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f} @ Epoch {best_epoch})')
            print(f'          Test Accuracy: {test_accuracy:.4f} (Best: {best_test_accuracy:.4f} @ Epoch {best_test_epoch})')
            
            # Log to tensorboard
            writer.add_scalar('Val/Accuracy', accuracy, epoch)
            writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
            writer.add_scalar('Train/InstanceAccuracy', train_instance_accuracy, epoch)
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
    
    writer.close()
    total_time = time.time() - start_time
    print(f'Training completed.')
    print(f'Best validation accuracy: {best_accuracy:.4f} (achieved at epoch {best_epoch})')
    print(f'Best test accuracy: {best_test_accuracy:.4f} (achieved at epoch {best_test_epoch})')
    print(f'Best train instance accuracy: {best_train_instance_accuracy:.4f} (achieved at epoch {best_train_instance_epoch})')
    
    
    print(f'Total training time: {format_elapsed_time(total_time)}')
    
    return model, best_accuracy


if __name__ == '__main__':
    # Default configuration
    config = {
        'patch_size': 4,
        'embed_dim': 384,
        'num_heads': 6,
        'L': 6,
        'bag_size': 5,
        'mini_batch_size': 8,
        'num_classes': 10,
        'epochs': 100,
        'learning_rate': 1e-4,
        'eval_interval': 5,
        'data_root': './data',
        'dropout': 0.1,
        'weight_decay': 0.01,
        'mlp_ratio': 4.0,
        'grad_clip': None,
        'warmup_epochs': 0,
        'valid_ratio': 0.1,
        'seed': 42
    }
    
    train(config)