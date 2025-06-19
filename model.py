import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvolutionalTokenizer(nn.Module):
    """Convert image to patch embeddings using Convolutional layers."""
    
    def __init__(self, img_size=32, in_channels=3, embed_dim=768, 
                 conv_layers=2, kernel_size=3, stride=1, padding=1, 
                 pool_kernel=2, pool_stride=2):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        
        # Build convolutional tokenizer
        layers = []
        current_channels = in_channels
        current_size = img_size
        
        for i in range(conv_layers):
            # Convolutional layer
            layers.append(nn.Conv2d(current_channels, embed_dim if i == conv_layers-1 else embed_dim//2, 
                                   kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            
            # Max pooling (except for the last layer to preserve more spatial information)
            if i < conv_layers - 1:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
                current_size = current_size // pool_stride
                current_channels = embed_dim // 2
            else:
                # Final max pooling with smaller stride to get appropriate token size
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
                current_size = current_size // pool_stride
        
        self.tokenizer = nn.Sequential(*layers)
        
        # Calculate final spatial dimensions
        self.final_size = current_size
        self.num_patches = current_size ** 2
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        
        # Apply convolutional tokenizer
        x = self.tokenizer(x)  # (B, embed_dim, final_size, final_size)
        
        # Reshape to sequence format
        x = x.flatten(2).contiguous()  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2).contiguous()  # (B, num_patches, embed_dim)
        
        return x


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings using Linear Projection (Legacy)."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        
        # Linear projection
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        
        # Reshape into patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # x: (B, C, H/patch_size, W/patch_size, patch_size, patch_size)
        
        x = x.contiguous().view(B, C, self.num_patches, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, num_patches, C, patch_size, patch_size)
        x = x.view(B, self.num_patches, self.patch_dim)  # (B, num_patches, C*patch_size*patch_size)
        
        # Linear projection
        x = self.proj(x)  # (B, num_patches, embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Expand mask to match attention dimensions
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with Global and Local Attention."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.global_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.local_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Pre-computed local attention masks for different configurations
        self.local_masks = {}
    
    def _get_or_create_local_mask(self, num_images, num_patches_per_image, device):
        """Get or create a pre-computed local attention mask."""
        key = (num_images, num_patches_per_image, device)
        
        if key not in self.local_masks:
            total_patches = num_images * num_patches_per_image
            mask = torch.zeros(total_patches, total_patches, device=device)
            
            for i in range(num_images):
                start_idx = i * num_patches_per_image
                end_idx = (i + 1) * num_patches_per_image
                mask[start_idx:end_idx, start_idx:end_idx] = 1
            
            self.local_masks[key] = mask
        
        return self.local_masks[key]
        
    def forward(self, x, num_patches_per_image):
        # Global attention across all tokens
        x = x + self.global_attn(self.norm1(x))
        
        # Local attention within each image
        B, N, C = x.shape
        bag_cls_token = x[:, 0:1, :]  # (B, 1, C)
        patch_tokens = x[:, 1:, :]  # (B, N-1, C)
        
        # Get pre-computed local attention mask
        num_images = (N - 1) // num_patches_per_image
        local_mask = self._get_or_create_local_mask(num_images, num_patches_per_image, x.device)
        
        # Expand mask for batch dimension
        local_mask = local_mask.unsqueeze(0).expand(B, -1, -1)
        
        # Apply local attention
        patch_tokens = patch_tokens + self.local_attn(self.norm2(patch_tokens), mask=local_mask)
        
        # Concatenate back
        x = torch.cat([bag_cls_token, patch_tokens], dim=1)
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x


class LLPAttentionModel(nn.Module):
    """Learning From Label Proportions with Attention model."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, dropout=0.1,
                 use_conv_tokenizer=True, conv_layers=2, kernel_size=3):
        super().__init__()
        
        if use_conv_tokenizer:
            self.patch_embed = ConvolutionalTokenizer(
                img_size=img_size, 
                in_channels=in_channels, 
                embed_dim=embed_dim,
                conv_layers=conv_layers,
                kernel_size=kernel_size
            )
        else:
            self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        self.num_patches = self.patch_embed.num_patches
        
        # BAG_CLS token
        self.bag_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (optional for convolutional tokenizer)
        self.use_pos_embed = not use_conv_tokenizer  # Disable for conv tokenizer by default
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        else:
            self.register_parameter('pos_embed', None)
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.bag_cls_token, std=0.02)
        if self.use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # x: (batch_size, num_images, channels, height, width)
        B, num_images, C, H, W = x.shape
        
        # Flatten batch and num_images dimensions
        x = x.view(B * num_images, C, H, W)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B * num_images, num_patches, embed_dim)
        
        # Reshape back
        x = x.reshape(B, num_images * self.num_patches, -1)
        
        # Add BAG_CLS token
        bag_cls_tokens = self.bag_cls_token.expand(B, -1, -1)
        x = torch.cat([bag_cls_tokens, x], dim=1)  # (B, 1 + num_images * num_patches, embed_dim)
        
        # Add positional embeddings (if enabled)
        if self.use_pos_embed:
            # For simplicity, we tile the positional embeddings for each image
            pos_embed = self.pos_embed[:, 1:, :].repeat(1, num_images, 1)
            pos_embed = torch.cat([self.pos_embed[:, :1, :], pos_embed], dim=1)
            x = x + pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, self.num_patches)
            
        x = self.norm(x)
        
        # Extract BAG_CLS token and classify
        bag_cls_output = x[:, 0]
        logits = self.head(bag_cls_output)
        
        return logits