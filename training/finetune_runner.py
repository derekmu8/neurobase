"""
Fine-tuning loop for seizure classification.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Dict
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import EEGEncoder
from models.classifier import SeizureClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class FinetuneRunner:
    """
    Handles fine-tuning loop for seizure classification.
    """
    
    def __init__(
        self,
        model: SeizureClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: str = 'mps',
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function with class weights for imbalanced data
        if class_weights is not None:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.training.finetune_lr,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.finetune_epochs,
            eta_min=1e-6
        )
        
        # Tracking
        self.train_losses = []
        self.val_metrics = []
        self.best_val_f1 = 0
        self.best_val_auroc = 0
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in pbar:
            x = batch['signal'].to(self.device)
            y = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output['logits'], y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip
            )
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        self.scheduler.step()
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate and compute metrics"""
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            x = batch['signal'].to(self.device)
            y = batch['label'].to(self.device)
            
            output = self.model(x)
            loss = self.criterion(output['logits'], y)
            
            total_loss += loss.item()
            n_batches += 1
            
            probs = output['probs'].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auroc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, path: Path, is_best: bool = False):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.model.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'best_val_f1': self.best_val_f1,
            'best_val_auroc': self.best_val_auroc,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / 'best_classifier.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_val_auroc = checkpoint['best_val_auroc']
        
        return checkpoint['epoch']
    
    def train(
        self,
        n_epochs: int,
        checkpoint_dir: Path,
        save_every: int = 5,
        unfreeze_after: int = 5
    ):
        """
        Full training loop.
        
        Args:
            n_epochs: Number of epochs
            checkpoint_dir: Where to save checkpoints
            save_every: Save every N epochs
            unfreeze_after: Unfreeze encoder after N epochs
        """
        print(f"\nStarting fine-tuning for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Encoder frozen: {self.model.freeze_encoder}")
        print()
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Unfreeze encoder after warmup
            if epoch == unfreeze_after and self.model.freeze_encoder:
                print(f"\nUnfreezing encoder at epoch {epoch+1}")
                self.model.unfreeze_encoder()
                # Reset optimizer with lower LR for encoder
                self.optimizer = AdamW([
                    {'params': self.model.encoder.parameters(), 'lr': self.config.training.finetune_lr * 0.1},
                    {'params': self.model.classifier.parameters(), 'lr': self.config.training.finetune_lr}
                ], weight_decay=self.config.training.weight_decay)
            
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Check if best
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_auroc = val_metrics['auroc']
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(
                    epoch,
                    checkpoint_dir / f'finetune_epoch_{epoch+1}.pt',
                    is_best=is_best
                )
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f} {'(best)' if is_best else ''}")
            print(f"  Val AUROC: {val_metrics['auroc']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nFine-tuning complete in {total_time/60:.1f} minutes")
        print(f"Best F1: {self.best_val_f1:.4f}")
        print(f"Best AUROC: {self.best_val_auroc:.4f}")
        
        return self.train_losses, self.val_metrics


def run_finetuning(
    encoder: EEGEncoder,
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    freeze_encoder: bool = True,
    resume_from: Optional[Path] = None,
    class_weights: Optional[torch.Tensor] = None
) -> SeizureClassifier:
    """
    Main function to run fine-tuning.
    
    Args:
        encoder: Pretrained encoder
        config: Config object
        train_loader: Training data loader
        val_loader: Validation data loader
        freeze_encoder: Whether to freeze encoder initially
        resume_from: Optional checkpoint path
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Trained classifier
    """
    # Create classifier
    classifier = SeizureClassifier(
        encoder=encoder,
        freeze_encoder=freeze_encoder
    )
    
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")
    
    # Create runner
    runner = FinetuneRunner(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config.device,
        class_weights=class_weights
    )
    
    # Resume if specified
    if resume_from and resume_from.exists():
        start_epoch = runner.load_checkpoint(resume_from)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    checkpoint_dir = Path(config.training.checkpoint_dir)
    runner.train(
        n_epochs=config.training.finetune_epochs,
        checkpoint_dir=checkpoint_dir,
        save_every=config.training.save_every,
        unfreeze_after=5
    )
    
    return classifier


if __name__ == "__main__":
    # Test with dummy data
    from config import config
    from models.encoder import create_encoder_from_config
    from torch.utils.data import DataLoader
    
    # Dummy data
    class DummyDataset:
        def __init__(self, n_samples):
            self.n = n_samples
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            return {
                'signal': torch.randn(config.model.n_channels, config.model.window_samples),
                'label': torch.tensor(float(idx % 2))
            }
    
    train_loader = DataLoader(DummyDataset(100), batch_size=16, shuffle=True)
    val_loader = DataLoader(DummyDataset(20), batch_size=16)
    
    # Create encoder and run finetuning
    encoder = create_encoder_from_config(config)
    config.training.finetune_epochs = 2
    
    classifier = run_finetuning(encoder, config, train_loader, val_loader)
    print("Fine-tuning test complete!")
