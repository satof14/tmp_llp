import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import LLPAttentionModel
from dataset import get_single_image_dataloader


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
    model = LLPAttentionModel(
        img_size=32,
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['L'],
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataloader
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
        for images, labels in tqdm(test_loader, desc='Evaluating'):
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
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
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
    plt.savefig('confusion_matrix.png')
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


def analyze_predictions(results, num_examples=10):
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
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        for i in range(min(num_examples, len(misclassified_indices))):
            idx = misclassified_indices[i]
            true_label = labels[idx]
            pred_label = predictions[idx]
            confidence = torch.softmax(logits[idx], dim=0)[pred_label].item()
            
            print(f'  Example {i+1}: True: {class_names[true_label]}, '
                  f'Predicted: {class_names[pred_label]} (confidence: {confidence:.3f})')


if __name__ == '__main__':
    # Evaluate the best model
    results = evaluate_model('best_model.pth')
    
    # Analyze predictions
    analyze_predictions(results)