import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import time

from model import LLPAttentionModel
from dataset import get_bag_dataloader, get_single_image_dataloader, get_mifcm_bag_dataloader, get_mifcm_single_image_dataloader, get_human_somatic_small_bag_dataloader, get_human_somatic_small_single_image_dataloader


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
    
    # Create dataloaders
    if config.get('dataset') == 'mifcm_3classes_newgate':
        train_loader = get_mifcm_bag_dataloader(
            root=config['data_root'],
            train=True,
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True
        )
        
        val_loader = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            train=False,
            batch_size=100,
            shuffle=False
        )
        
        # Create train instance-level dataloader for train accuracy evaluation
        train_instance_loader = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            train=True,
            batch_size=100,
            shuffle=False
        )
    elif config.get('dataset') == 'human_somatic_small':
        train_loader = get_human_somatic_small_bag_dataloader(
            root=config['data_root'],
            split='train',
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True
        )
        
        val_loader = get_human_somatic_small_single_image_dataloader(
            root=config['data_root'],
            split='test',
            batch_size=100,
            shuffle=False
        )
        
        # Create train instance-level dataloader for train accuracy evaluation
        train_instance_loader = get_human_somatic_small_single_image_dataloader(
            root=config['data_root'],
            split='train',
            batch_size=100,
            shuffle=False
        )
    else:
        train_loader = get_bag_dataloader(
            root=config['data_root'],
            train=True,
            bag_size=config['bag_size'],
            batch_size=config['mini_batch_size'],
            shuffle=True
        )
        
        val_loader = get_single_image_dataloader(
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
    
    # Create subset of training data to match validation set size (10,000 samples)
    val_size = len(val_loader.dataset)
    if len(train_instance_loader.dataset) > val_size:
        indices = torch.randperm(len(train_instance_loader.dataset))[:val_size]
        subset_dataset = Subset(train_instance_loader.dataset, indices)
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
    
    # Create scheduler
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
            
            # Evaluate on training set (instance-level)
            train_instance_accuracy = evaluate(model, train_instance_loader, device)
            
            # Track best train instance accuracy
            if train_instance_accuracy > best_train_instance_accuracy:
                best_train_instance_accuracy = train_instance_accuracy
                best_train_instance_epoch = epoch
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'config': config
                }, os.path.join(log_dir, 'best_model.pth') if log_dir else 'best_model.pth')
                print(f'Saved best model with accuracy: {accuracy:.4f}')
            
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch}/{config["epochs"]-1} | Elapsed: {format_elapsed_time(elapsed_time)}')
            print(f'Train Instance Accuracy: {train_instance_accuracy:.4f} (Best: {best_train_instance_accuracy:.4f} @ Epoch {best_train_instance_epoch})')
            print(f'           Val Accuracy: {accuracy:.4f} (Best: {best_accuracy:.4f} @ Epoch {best_epoch})')
            
            # Log to tensorboard
            writer.add_scalar('Val/Accuracy', accuracy, epoch)
            writer.add_scalar('Train/InstanceAccuracy', train_instance_accuracy, epoch)
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
    
    writer.close()
    total_time = time.time() - start_time
    print(f'Training completed.')
    print(f'Best validation accuracy: {best_accuracy:.4f} (achieved at epoch {best_epoch})')
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
        'grad_clip': None
    }
    
    train(config)