import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Learning From Label Proportions with Attention')
    
    # Model architecture arguments
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Size of image patches')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='Token embedding dimension')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads')
    parser.add_argument('--L', type=int, default=12,
                        help='Number of encoder layers')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP hidden dimension ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--patch_embed_type', type=str, default='linear',
                        choices=['linear', 'cct'],
                        help='Type of patch embedding: linear or cct')
    
    # Training arguments
    parser.add_argument('--bag_size', type=int, default=8,
                        help='Fixed bag size during training')
    parser.add_argument('--mini_batch_size', type=int, default=2,
                        help='Bag-level batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Model checkpoint save interval in epochs')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval in batches')
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam/AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam/AdamW optimizer')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for optimizer')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mifcm_3classes_newgate', 'human_somatic_small'],
                        help='Dataset to use')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Experiment arguments
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to model checkpoint for evaluation')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--exp_name', type=str, default='llp_attention',
                        help='Experiment name')
    parser.add_argument('--save_best', action='store_true',
                        help='Save best model based on validation accuracy')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
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