import torch
import os
import json
import sys
import uuid
from datetime import datetime

from args import get_args
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
    # Get arguments from args.py
    args = get_args()
    
    # Create timestamped directory for logs
    timestamp = datetime.now().strftime("%Y%m%dT%H_%M_%S")
    run_id = str(uuid.uuid4())[:4]  # Use first 4 characters of UUID
    log_dir = f"results/llp_attention_{timestamp}_{run_id}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Redirect all output to both console and log file
    log_file = open(os.path.join(log_dir, 'all.log'), 'w')
    sys.stdout = TeeOutput(sys.stdout, log_file)
    sys.stderr = TeeOutput(sys.stderr, log_file)
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Auto-detect num_classes if not specified
    if args.num_classes is None:
        if args.dataset == 'mifcm_3classes_newgate':
            args.num_classes = 3
        elif args.dataset == 'human_somatic_small':
            args.num_classes = 3
        else:  # cifar10
            args.num_classes = 10
    
    # Create config dictionary
    config = {
        'dataset': args.dataset,
        'patch_size': args.patch_size,
        'embed_dim': args.embed_dim,
        'num_heads': args.num_heads,
        'L': args.L,
        'mlp_ratio': args.mlp_ratio,
        'bag_size': args.bag_size,
        'mini_batch_size': args.mini_batch_size,
        'num_classes': args.num_classes,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'eval_interval': args.eval_interval,
        'grad_clip': args.grad_clip,
        'warmup_epochs': args.warmup_epochs,
        'data_root': args.data_root,
        'seed': args.seed,
        'valid_ratio': args.valid_ratio,
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'eps': args.eps,
        'patch_embed_type': args.patch_embed_type
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