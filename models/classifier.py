"""
Classification head for seizure detection fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import EEGEncoder


class SeizureClassifier(nn.Module):
    """
    Binary classifier for seizure detection.
    
    Takes pretrained encoder, adds classification head on CLS token.
    Supports freezing encoder during initial fine-tuning.
    """
    
    def __init__(
        self,
        encoder: EEGEncoder,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(encoder.d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Binary classification
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> dict:
        """
        Args:
            x: (batch, n_channels, n_samples)
            return_attention: Whether to return attention for visualization
        
        Returns:
            dict with logits, probabilities, and optionally attention
        """
        # Get encoder output
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_out = self.encoder(x, return_attention=return_attention)
        else:
            encoder_out = self.encoder(x, return_attention=return_attention)
        
        # Classify from CLS embedding
        cls_embedding = encoder_out['cls_embedding']
        logits = self.classifier(cls_embedding).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probs': probs,
            'cls_embedding': cls_embedding,
            'attention_weights': encoder_out.get('attention_weights')
        }
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning"""
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get averaged attention weights for visualization.
        
        Returns:
            (batch, n_patches) attention from CLS to patches
        """
        with torch.no_grad():
            output = self.forward(x, return_attention=True)
        
        if output['attention_weights'] is None:
            return None
        
        # Average attention across heads and layers
        # Each attention is (batch, n_heads, seq_len, seq_len)
        # We want attention from CLS (position 0) to patches
        
        attn_weights = []
        for layer_attn in output['attention_weights']:
            # Get CLS attention to all positions: (batch, n_heads, seq_len)
            cls_attn = layer_attn[:, :, 0, 1:]  # Exclude CLS-to-CLS
            # Average over heads
            cls_attn = cls_attn.mean(dim=1)  # (batch, n_patches)
            attn_weights.append(cls_attn)
        
        # Average over layers
        avg_attn = torch.stack(attn_weights).mean(dim=0)
        
        return avg_attn


class BaselineCNN(nn.Module):
    """
    Simple CNN baseline for comparison.
    """
    
    def __init__(
        self,
        n_channels: int = 18,
        n_samples: int = 1024
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, n_channels, n_samples)
        
        Returns:
            dict with logits and probs
        """
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten and classify
        x = x.squeeze(-1)
        logits = self.fc(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probs': probs
        }


def create_classifier(
    encoder: EEGEncoder,
    freeze_encoder: bool = True,
    **kwargs
) -> SeizureClassifier:
    """Factory function to create classifier"""
    return SeizureClassifier(
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        **kwargs
    )


if __name__ == "__main__":
    from .encoder import EEGEncoder
    
    # Test classifier
    encoder = EEGEncoder()
    classifier = SeizureClassifier(encoder, freeze_encoder=True)
    
    x = torch.randn(4, 18, 1024)
    output = classifier(x, return_attention=True)
    
    print(f"Logits: {output['logits'].shape}")
    print(f"Probs: {output['probs']}")
    
    # Get attention for visualization
    attn = classifier.get_attention_weights(x)
    print(f"Attention: {attn.shape}")
    
    # Test baseline CNN
    baseline = BaselineCNN()
    output = baseline(x)
    print(f"Baseline output: {output['probs']}")
