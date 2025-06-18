import argparse
import torch
import os
import json
import sys
import uuid
from datetime import datetime

from train import train
from evaluate import evaluate_model


class TeeOutput:
    def __init__(self, *files):
        self.files = files
        # 元のstdoutの属性を保持
        self._isatty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()
    
    def flush(self):
        for file in self.files:
            file.flush()
    
    def isatty(self):
        # tqdmが端末かどうかを判定するために使用
        return self._isatty
    
    def fileno(self):
        # ファイルディスクリプタを返す
        return self.files[0].fileno() if self.files else 1
    
    def __getattr__(self, name):
        # その他の属性は最初のファイルから取得
        return getattr(self.files[0], name) if self.files else None

def main():
    # Create timestamped directory for logs
    timestamp = datetime.now().strftime("%Y%m%dT%H_%M_%S")
    run_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    log_dir = f"results/llp_attention_{timestamp}_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Redirect all output to both console and log file
    log_file = open(os.path.join(log_dir, 'all.log'), 'w')
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)
    
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
    
    # Training arguments
    parser.add_argument('--bag_size', type=int, default=1,
                        help='Size of bags for training (default: 1)')
    parser.add_argument('--mini_batch_size', type=int, default=50,
                        help='Batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='Evaluation interval in epochs (default: 1)')
    
    # Other arguments
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes (default: 10 for CIFAR-10)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets (default: ./data)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or eval (default: train)')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to model checkpoint for evaluation (default: best_model.pth)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Create config dictionary
    config = {
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'L': args.L,
        'bag_size': args.bag_size,
        'mini_batch_size': args.mini_batch_size,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'eval_interval': args.eval_interval,
        'data_root': args.data_root,
        'seed': args.seed
    }
    
    if args.mode == 'train':
        print('Starting training...')
        print('Configuration:')
        for key, value in config.items():
            print(f'  {key}: {value}')
        
        # Save config
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Train model
        model, best_accuracy = train(config, log_dir)
        print(f'Training completed. Best accuracy: {best_accuracy:.4f}')
        
    elif args.mode == 'eval':
        print('Starting evaluation...')
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f'Model checkpoint not found: {args.model_path}')
        
        # Evaluate model
        results = evaluate_model(args.model_path)
        print(f'Evaluation completed. Test accuracy: {results["accuracy"]:.4f}')


if __name__ == '__main__':
    main()