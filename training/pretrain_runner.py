"""
Pretraining loop for masked patch prediction.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import EEGEncoder, create_encoder_from_config
from models.pretrain import MaskedPatchPredictor


class PretrainRunner:
    """
    Handles pretraining loop with logging and checkpointing.
    """
    
    def __init__(
        self,
        model: MaskedPatchPredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'mps'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.pretrain_lr,
            weight_decay=config.training.weight_decay
        )
        
        # Learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.warmup_steps
        )
        
        main_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(train_loader) * 10,  # Restart every 10 epochs
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.training.warmup_steps]
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            # Move to device
            if isinstance(batch, dict):
                x = batch['signal'].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.grad_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Track
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            if isinstance(batch, dict):
                x = batch['signal'].to(self.device)
            else:
                x = batch.to(self.device)
            
            output = self.model(x)
            total_loss += output['loss'].item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def save_checkpoint(self, epoch: int, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / 'best_pretrain.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        return checkpoint['epoch']
    
    def train(
        self, 
        n_epochs: int,
        checkpoint_dir: Path,
        save_every: int = 5
    ):
        """Full training loop"""
        print(f"\nStarting pretraining for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Check if best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(
                    epoch, 
                    checkpoint_dir / f'pretrain_epoch_{epoch+1}.pt',
                    is_best=is_best
                )
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f} {'(best)' if is_best else ''}")
            print(f"  Time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nPretraining complete in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses


def run_pretraining(config, train_loader, val_loader, resume_from: Optional[Path] = None):
    """
    Main function to run pretraining.
    
    Args:
        config: Config object
        train_loader: Training data loader
        val_loader: Validation data loader
        resume_from: Optional checkpoint path to resume from
    
    Returns:
        Trained encoder
    """
    # Create model
    encoder = create_encoder_from_config(config)
    pretrainer = MaskedPatchPredictor(
        encoder=encoder,
        mask_ratio=config.model.mask_ratio
    )
    
    print(f"Model parameters: {encoder.get_num_params():,}")
    
    # Create runner
    runner = PretrainRunner(
        model=pretrainer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device
    )
    
    # Resume if specified
    start_epoch = 0
    if resume_from and resume_from.exists():
        start_epoch = runner.load_checkpoint(resume_from)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    checkpoint_dir = Path(config.training.checkpoint_dir)
    runner.train(
        n_epochs=config.training.pretrain_epochs,
        checkpoint_dir=checkpoint_dir,
        save_every=config.training.save_every
    )
    
    return encoder


if __name__ == "__main__":
    # Test with dummy data
    from config import config
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data
    n_train = 100
    n_val = 20
    
    train_data = torch.randn(n_train, config.model.n_channels, config.model.window_samples)
    val_data = torch.randn(n_val, config.model.n_channels, config.model.window_samples)
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=16)
    
    # Quick test
    config.training.pretrain_epochs = 2
    encoder = run_pretraining(config, train_loader, val_loader)
    print("Pretraining test complete!")
