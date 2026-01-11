"""
Configuration for NeuroBase EEG Foundation Model
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class DataConfig:
    """Data pipeline configuration"""
    # CHB-MIT dataset settings
    data_dir: Path = Path("./data/chbmit")
    train_patients: List[str] = field(default_factory=lambda: ["chb01", "chb02"])
    val_patients: List[str] = field(default_factory=lambda: ["chb03"])
    test_patients: List[str] = field(default_factory=lambda: ["chb03"])  # Same as val for now
    
    # Signal parameters
    sfreq: int = 256  # Sampling frequency (Hz)
    window_sec: float = 4.0  # Window size in seconds
    window_samples: int = 1024  # window_sec * sfreq
    stride_sec: float = 1.0  # Stride for sliding window
    
    # Preprocessing
    l_freq: float = 0.5  # High-pass filter
    h_freq: float = 100.0  # Low-pass filter
    n_channels: int = 18  # Common channels across patients
    
    # Common channel names in CHB-MIT (subset that's consistent)
    common_channels: List[str] = field(default_factory=lambda: [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ'
    ])


@dataclass
class ModelConfig:
    """Transformer encoder configuration"""
    # Input
    n_channels: int = 18
    window_samples: int = 1024
    
    # Patch embedding (treat time as patches)
    patch_size: int = 64  # 64 samples = 0.25 sec at 256Hz
    n_patches: int = 16  # window_samples // patch_size
    
    # Transformer
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    
    # Pretraining
    mask_ratio: float = 0.4  # Fraction of patches to mask


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Pretraining
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 64
    pretrain_lr: float = 1e-4
    
    # Fine-tuning
    finetune_epochs: int = 30
    finetune_batch_size: int = 32
    finetune_lr: float = 5e-5
    
    # General
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    
    # Checkpoints
    checkpoint_dir: Path = Path("./checkpoints")
    save_every: int = 5


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Device
    device: str = "mps"  # M2 Pro Mac
    seed: int = 42


# Global config instance
config = Config()
