import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Learning From Label Proportions with Attention')
    
    # Model arguments
    parser.add_argument('--patch_size', type=int, default=8,
                        help='Size of image patches (default: 8)')
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='Embedding dimension (default: 384)')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of attention heads (default: 6)')
    parser.add_argument('--L', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP ratio for transformer blocks (default: 4.0)')
    parser.add_argument('--patch_embed_type', type=str, default='cct',
                        help='Type of patch embedding: "cct" for Compact Convolutional Transformer or "linear" for Linear Projection (default: cct)')
    
    # Training arguments
    parser.add_argument('--bag_size', type=int, default=1,
                        help='Size of bags for training (default: 1)')
    parser.add_argument('--mini_batch_size', type=int, default=50,
                        help='Batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer (default: 0.01)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval in epochs (default: 1)')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient clipping max norm (default: None, no clipping)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs (default: 0, no warmup)')
    
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
    
    # Other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mifcm_3classes_newgate', 'human_somatic_small'],
                        help='Dataset to use (default: cifar10)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (auto-detected based on dataset if not specified)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets (default: ./data)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval (default: train)')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to model checkpoint for evaluation (default: best_model.pth)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    
    args = parser.parse_args()
    return args