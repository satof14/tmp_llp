# LLP Attention: Learning From Label Proportions with Attention

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python main.py --mode train --epochs 100 --mini_batch_size 8
```

## Evaluation

```bash
python main.py --mode eval --model_path best_model.pth
```

## Key Arguments

- `--patch_size`: Image patch size (default: 4)
- `--embed_dim`: Embedding dimension (default: 384)
- `--num_heads`: Number of attention heads (default: 6)
- `--L`: Number of transformer layers (default: 6)
- `--bag_size`: Training bag size (default: 5)
- `--mini_batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 1e-4)

## Output

- `best_model.pth`: Best model checkpoint
- `runs/llp_attention/`: TensorBoard logs
- `confusion_matrix.png`: Confusion matrix visualization