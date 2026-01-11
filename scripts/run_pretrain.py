#!/usr/bin/env python
"""
Run pretraining on CHB-MIT dataset.

Usage:
    python scripts/run_pretrain.py
    python scripts/run_pretrain.py --epochs 100 --batch-size 32
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data import create_dataloaders
from training import run_pretraining


def main():
    parser = argparse.ArgumentParser(description="Pretrain EEG encoder")
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (mps, cuda, cpu)')
    args = parser.parse_args()
    
    # Override config with command line args
    if args.epochs:
        config.training.pretrain_epochs = args.epochs
    if args.batch_size:
        config.training.pretrain_batch_size = args.batch_size
    if args.lr:
        config.training.pretrain_lr = args.lr
    if args.device:
        config.device = args.device
    
    print("="*60)
    print("NEUROBASE PRETRAINING")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.pretrain_epochs}")
    print(f"Batch size: {config.training.pretrain_batch_size}")
    print(f"Learning rate: {config.training.pretrain_lr}")
    print(f"Train patients: {config.data.train_patients}")
    print(f"Val patients: {config.data.val_patients}")
    print("="*60)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=config.data.data_dir,
        train_patients=config.data.train_patients,
        val_patients=config.data.val_patients,
        test_patients=config.data.test_patients,
        channels=config.data.common_channels,
        config=config,
        mode='pretrain'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Run pretraining
    resume_path = Path(args.resume) if args.resume else None
    encoder = run_pretraining(config, train_loader, val_loader, resume_from=resume_path)
    
    print("\nPretraining complete!")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
