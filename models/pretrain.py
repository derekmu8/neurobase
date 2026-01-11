"""
Self-supervised pretraining with masked patch prediction.

Similar to BERT/MAE: mask random patches, predict their content.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from .encoder import EEGEncoder


class MaskedPatchPredictor(nn.Module):
    """
    Pretraining model with masked patch prediction.
    
    1. Embed patches
    2. Mask random patches (replace with learnable mask token)
    3. Encode with transformer
    4. Predict original patch content for masked positions
    """
    
    def __init__(
        self,
        encoder: EEGEncoder,
        mask_ratio: float = 0.4
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # Learnable mask token
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, encoder.d_model) * 0.02
        )
        
        # Decoder to predict original patch content
        # Predicts (n_channels * patch_size) values per patch
        patch_dim = encoder.n_channels * encoder.patch_size
        self.decoder = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.GELU(),
            nn.Linear(encoder.d_model, patch_dim)
        )
    
    def random_mask(
        self, 
        batch_size: int, 
        n_patches: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate random mask for patches.
        
        Returns:
            mask: (batch, n_patches) boolean tensor, True = masked
        """
        n_mask = int(n_patches * self.mask_ratio)
        
        # Random permutation for each sample
        noise = torch.rand(batch_size, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Create mask
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_mask], True)
        
        return mask
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            x: (batch, n_channels, n_samples) raw EEG signal
            mask: Optional pre-defined mask
        
        Returns:
            dict with loss, predictions, targets, mask
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get original patches for reconstruction target
        # (batch, n_channels, n_patches, patch_size)
        x_patches = x.unfold(2, self.encoder.patch_size, self.encoder.patch_size)
        # (batch, n_patches, n_channels, patch_size)
        x_patches = x_patches.permute(0, 2, 1, 3)
        # (batch, n_patches, n_channels * patch_size)
        targets = x_patches.reshape(batch_size, self.encoder.n_patches, -1)
        
        # Generate mask if not provided
        if mask is None:
            mask = self.random_mask(batch_size, self.encoder.n_patches, device)
        
        # Embed patches (without CLS for now)
        embeddings = self.encoder.patch_embed(x, include_cls=False)
        
        # Replace masked patches with mask token
        mask_tokens = self.mask_token.expand(batch_size, self.encoder.n_patches, -1)
        embeddings = torch.where(
            mask.unsqueeze(-1).expand_as(embeddings),
            mask_tokens,
            embeddings
        )
        
        # Add CLS token
        cls_tokens = self.encoder.patch_embed.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Create attention mask (CLS + patches)
        # We don't mask in attention - model sees all positions but must predict masked content
        
        # Pass through transformer
        for block in self.encoder.blocks:
            embeddings, _ = block(embeddings)
        
        embeddings = self.encoder.norm(embeddings)
        
        # Get patch embeddings (exclude CLS)
        patch_embeddings = embeddings[:, 1:]
        
        # Predict original patches
        predictions = self.decoder(patch_embeddings)
        
        # Compute loss only on masked patches
        loss = self._masked_mse_loss(predictions, targets, mask)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'mask': mask,
            'cls_embedding': embeddings[:, 0]
        }
    
    def _masked_mse_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss computed only on masked positions"""
        # (batch, n_patches, patch_dim)
        loss = (predictions - targets) ** 2
        loss = loss.mean(dim=-1)  # Average over patch dimension
        
        # Only count masked positions
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        return loss


class ContrastivePretrainer(nn.Module):
    """
    Alternative: Contrastive learning pretraining.
    
    Create positive pairs from same signal with augmentation,
    negative pairs from different signals.
    """
    
    def __init__(
        self,
        encoder: EEGEncoder,
        projection_dim: int = 128,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model),
            nn.ReLU(),
            nn.Linear(encoder.d_model, projection_dim)
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
        """
        Args:
            x1, x2: Two augmented views of same signals
                    (batch, n_channels, n_samples)
        
        Returns:
            dict with loss and embeddings
        """
        # Encode both views
        z1 = self.encoder(x1)['cls_embedding']
        z2 = self.encoder(x2)['cls_embedding']
        
        # Project
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        
        # Normalize
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        
        # InfoNCE loss
        loss = self._info_nce_loss(p1, p2)
        
        return {
            'loss': loss,
            'embedding1': z1,
            'embedding2': z2
        }
    
    def _info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE contrastive loss"""
        batch_size = z1.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)
        
        # Create labels: positive pairs are (i, i+batch) and (i+batch, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # (2*batch,)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


def create_pretrainer(encoder: EEGEncoder, method: str = 'masked', **kwargs):
    """
    Factory function to create pretraining model.
    
    Args:
        encoder: EEG encoder
        method: 'masked' or 'contrastive'
    """
    if method == 'masked':
        return MaskedPatchPredictor(encoder, **kwargs)
    elif method == 'contrastive':
        return ContrastivePretrainer(encoder, **kwargs)
    else:
        raise ValueError(f"Unknown pretraining method: {method}")


if __name__ == "__main__":
    from .encoder import EEGEncoder
    
    # Test masked prediction
    encoder = EEGEncoder()
    pretrainer = MaskedPatchPredictor(encoder, mask_ratio=0.4)
    
    x = torch.randn(4, 18, 1024)
    output = pretrainer(x)
    
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Predictions: {output['predictions'].shape}")
    print(f"Targets: {output['targets'].shape}")
    print(f"Mask: {output['mask'].shape}, {output['mask'].sum().item()} masked")
