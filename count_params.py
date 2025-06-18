from model import LLPAttentionModel

# Create model with default parameters
model = LLPAttentionModel(
    img_size=32,
    patch_size=8,  # Default from main.py
    in_channels=3,
    num_classes=10,
    embed_dim=384,
    num_heads=6,
    num_layers=6,  # L=6
    mlp_ratio=4.0,
    dropout=0.1
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters (M): {total_params/1e6:.2f}M")

# Breakdown by component
print("\nParameter breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params:,}")