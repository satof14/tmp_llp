import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import os
import json

from model import LLPAttentionModel
from dataset import get_single_image_dataloader, get_mifcm_single_image_dataloader


def evaluate_model(model_path, config=None, device=None):
    """Evaluate a trained model on single image classification."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config if not provided
    if config is None:
        config = checkpoint['config']
    
    # Create model
    img_size = 64 if config.get('dataset') == 'mifcm_3classes_newgate' else 32
    model = LLPAttentionModel(
        img_size=img_size,
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['L'],
        mlp_ratio=config['mlp_ratio'],
        dropout=0.1
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataloader
    if config.get('dataset') == 'mifcm_3classes_newgate':
        test_loader = get_mifcm_single_image_dataloader(
            root=config['data_root'],
            split='test',
            batch_size=100,
            shuffle=False
        )
    else:
        test_loader = get_single_image_dataloader(
            root=config['data_root'],
            train=False,
            batch_size=100,
            shuffle=False
        )
    
    # Evaluate
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating', file=sys.__stdout__, ncols=80):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    print(f'\nTest Accuracy: {accuracy:.4f}')
    
    # Classification report
    if config.get('dataset') == 'mifcm_3classes_newgate':
        class_names = ['G1', 'S', 'G2']
    else:
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck'][:config['num_classes']]
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, 
                              target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # Save confusion matrix to results directory
    cm_path = os.path.join(args.results_dir, 'confusion_matrix_eval.png')
    plt.savefig(cm_path)
    print(f'\nConfusion matrix saved to: {cm_path}')
    plt.close()
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(config['num_classes']):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_predictions[mask] == i).mean()
            per_class_acc.append(acc)
            print(f'{class_names[i]}: {acc:.4f}')
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm
    }


def analyze_predictions(results, config, num_examples=10):
    """Analyze model predictions and show examples of errors."""
    predictions = results['predictions']
    labels = results['labels']
    logits = results['logits']
    
    # Find misclassified examples
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) > 0:
        print(f'\nTotal misclassified: {len(misclassified_indices)} / {len(labels)}')
        
        # Show some examples
        print(f'\nShowing {min(num_examples, len(misclassified_indices))} misclassified examples:')
        if config.get('dataset') == 'mifcm_3classes_newgate':
            class_names = ['G1', 'S', 'G2']
        else:
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck'][:config['num_classes']]
        
        for i in range(min(num_examples, len(misclassified_indices))):
            idx = misclassified_indices[i]
            true_label = labels[idx]
            pred_label = predictions[idx]
            confidence = torch.softmax(logits[idx], dim=0)[pred_label].item()
            
            print(f'  Example {i+1}: True: {class_names[true_label]}, '
                  f'Predicted: {class_names[pred_label]} (confidence: {confidence:.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained LLP model')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the results directory containing the model checkpoint and config.json')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Name of the checkpoint file (default: best_model.pth)')
    args = parser.parse_args()
    
    # Load config.json from results directory
    config_path = os.path.join(args.results_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from: {config_path}")
    
    # Construct full model path
    model_path = os.path.join(args.results_dir, args.checkpoint)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)
    
    # Evaluate the model
    print(f"Evaluating model: {model_path}")
    results = evaluate_model(model_path, config=config)
    
    # Analyze predictions
    analyze_predictions(results, config)