"""
Transformer-based EEG Encoder

Architecture inspired by ViT and wav2vec 2.0, adapted for multi-channel EEG.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Convert EEG signal into patch embeddings.
    
    Input: (batch, n_channels, n_samples)
    Output: (batch, n_patches, d_model)
    
    Each patch covers all channels for a time segment.
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_samples: int = 1024,
        patch_size: int = 64,
        d_model: int = 256
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.n_patches = n_samples // patch_size
        self.d_model = d_model
        
        # Linear projection of flattened patches
        # Each patch is (n_channels, patch_size) -> flattened -> d_model
        self.proj = nn.Linear(n_channels * patch_size, d_model)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )
        
        # CLS token for classification tasks
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor, include_cls: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_samples)
            include_cls: Whether to prepend CLS token
        
        Returns:
            (batch, n_patches + 1, d_model) if include_cls
            (batch, n_patches, d_model) otherwise
        """
        batch_size = x.shape[0]
        
        # Reshape into patches: (batch, n_channels, n_patches, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)
        
        # (batch, n_patches, n_channels, patch_size)
        x = x.permute(0, 2, 1, 3)
        
        # Flatten patches: (batch, n_patches, n_channels * patch_size)
        x = x.reshape(batch_size, self.n_patches, -1)
        
        # Project to d_model
        x = self.proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        if include_cls:
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) boolean mask, True = masked
            return_attention: Whether to return attention weights
        
        Returns:
            output: (batch, seq_len, d_model)
            attention: (batch, n_heads, seq_len, seq_len) if return_attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for attention: (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Store for visualization
        self.attention_weights = attn.detach()
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn
        return out, None


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask, return_attention)
        x = x + attn_out
        
        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        
        return x, attn_weights


class EEGEncoder(nn.Module):
    """
    Transformer encoder for EEG signals.
    
    Takes multi-channel EEG windows and produces:
    - CLS embedding for classification
    - Patch embeddings for reconstruction/other tasks
    - Attention weights for visualization
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_samples: int = 1024,
        patch_size: int = 64,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = n_samples // patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_channels=n_channels,
            n_samples=n_samples,
            patch_size=patch_size,
            d_model=d_model
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> dict:
        """
        Args:
            x: (batch, n_channels, n_samples) raw EEG signal
            mask: (batch, n_patches) boolean mask for patches
            return_attention: Whether to return attention weights
        
        Returns:
            dict with:
                - cls_embedding: (batch, d_model)
                - patch_embeddings: (batch, n_patches, d_model)
                - attention_weights: list of (batch, n_heads, seq_len, seq_len)
        """
        # Get patch embeddings with CLS token
        x = self.patch_embed(x, include_cls=True)
        
        # Adjust mask for CLS token if provided
        if mask is not None:
            cls_mask = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask, return_attention)
            if return_attention:
                attention_weights.append(attn)
        
        # Final norm
        x = self.norm(x)
        
        # Split CLS and patch embeddings
        cls_embedding = x[:, 0]
        patch_embeddings = x[:, 1:]
        
        return {
            'cls_embedding': cls_embedding,
            'patch_embeddings': patch_embeddings,
            'attention_weights': attention_weights if return_attention else None
        }
    
    def get_num_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_encoder_from_config(config) -> EEGEncoder:
    """Create encoder from config object"""
    return EEGEncoder(
        n_channels=config.model.n_channels,
        n_samples=config.model.window_samples,
        patch_size=config.model.patch_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout
    )


if __name__ == "__main__":
    # Test encoder
    encoder = EEGEncoder(
        n_channels=18,
        n_samples=1024,
        patch_size=64,
        d_model=256,
        n_heads=4,
        n_layers=4
    )
    
    print(f"Model parameters: {encoder.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(4, 18, 1024)
    output = encoder(x, return_attention=True)
    
    print(f"CLS embedding: {output['cls_embedding'].shape}")
    print(f"Patch embeddings: {output['patch_embeddings'].shape}")
    print(f"Attention weights: {len(output['attention_weights'])} layers")
