import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Learning From Label Proportions with Attention')
    
    # Model architecture arguments
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Size of image patches (default: 4)')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='Token embedding dimension (default: 384)')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads (default: 6)')
    parser.add_argument('--L', type=int, default=12,
                        help='Number of encoder layers (default: 12)')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP hidden dimension ratio (default: 4.0)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--patch_embed_type', type=str, default='linear',
                        choices=['linear', 'cct'],
                        help='Type of patch embedding: linear or cct (default: linear)')
    
    # Training arguments
    parser.add_argument('--bag_size', type=int, default=8,
                        help='Fixed bag size during training (default: 8)')
    parser.add_argument('--mini_batch_size', type=int, default=2,
                        help='Bag-level batch size (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (default: 5)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval in epochs (default: 5)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Model checkpoint save interval in epochs (default: 10)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval in batches (default: 100)')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer (default: 0.9)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam/AdamW optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam/AdamW optimizer (default: 0.999)')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for optimizer (default: 1e-8)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mifcm_3classes_newgate', 'human_somatic_small'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes (default: 10 for CIFAR-10)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets (default: ./data)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    
    # Experiment arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Mode: train or eval (default: train)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to model checkpoint for evaluation (default: best_model.pth)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints (default: ./checkpoints)')
    parser.add_argument('--exp_name', type=str, default='llp_attention',
                        help='Experiment name (default: llp_attention)')
    parser.add_argument('--save_best', action='store_true',
                        help='Save best model based on validation accuracy')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training (default: cuda)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Auto-adjust num_classes based on dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'mifcm_3classes_newgate':
        args.num_classes = 3
    elif args.dataset == 'human_somatic_small':
        args.num_classes = 2
    
    return args