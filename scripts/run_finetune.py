#!/usr/bin/env python
"""
Run fine-tuning for seizure classification.

Usage:
    python scripts/run_finetune.py
    python scripts/run_finetune.py --pretrained checkpoints/best_pretrain.pt
"""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data import create_dataloaders
from models import create_encoder_from_config
from training import run_finetuning


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for seizure detection")
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--no-freeze', action='store_true', help='Do not freeze encoder')
    parser.add_argument('--device', type=str, default=None, help='Device (mps, cuda, cpu)')
    args = parser.parse_args()
    
    # Override config
    if args.epochs:
        config.training.finetune_epochs = args.epochs
    if args.batch_size:
        config.training.finetune_batch_size = args.batch_size
    if args.lr:
        config.training.finetune_lr = args.lr
    if args.device:
        config.device = args.device
    
    print("="*60)
    print("NEUROBASE FINE-TUNING")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.finetune_epochs}")
    print(f"Batch size: {config.training.finetune_batch_size}")
    print(f"Learning rate: {config.training.finetune_lr}")
    print(f"Pretrained: {args.pretrained or 'None (random init)'}")
    print(f"Freeze encoder: {not args.no_freeze}")
    print("="*60)
    
    # Create encoder
    encoder = create_encoder_from_config(config)
    
    # Load pretrained weights if provided
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            print(f"\nLoading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("Pretrained weights loaded successfully!")
        else:
            print(f"Warning: Pretrained checkpoint not found at {pretrained_path}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config.data.data_dir,
        train_patients=config.data.train_patients,
        val_patients=config.data.val_patients,
        test_patients=config.data.test_patients,
        channels=config.data.common_channels,
        config=config,
        mode='finetune'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Run fine-tuning
    classifier = run_finetuning(
        encoder=encoder,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        freeze_encoder=not args.no_freeze
    )
    
    # Final evaluation on test set
    if test_loader:
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        
        from evaluation import evaluate_model
        metrics, probs, labels = evaluate_model(
            classifier, 
            test_loader, 
            device=config.device
        )
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1: {metrics['f1']:.4f}")
        print(f"Test AUROC: {metrics['auroc']:.4f}")
    
    print("\nFine-tuning complete!")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
