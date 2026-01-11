#!/usr/bin/env python
"""
Evaluate models and compare against baselines.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --checkpoint checkpoints/best_classifier.pt
"""
import argparse
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from data import create_dataloaders, EEGDataset
from models import create_encoder_from_config, SeizureClassifier, BaselineCNN, RandomForestBaseline
from evaluation import (
    evaluate_model, evaluate_baseline, plot_roc_curve,
    print_comparison_table, generate_report
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate seizure detection models")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_classifier.pt',
                        help='Path to classifier checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    if args.device:
        config.device = args.device
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("NEUROBASE EVALUATION")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = EEGDataset(
        data_dir=config.data.data_dir,
        patients=config.data.test_patients,
        channels=config.data.common_channels,
        window_sec=config.data.window_sec,
        stride_sec=config.data.stride_sec,
        sfreq=config.data.sfreq,
        mode='finetune',
        balance_classes=False
    )
    
    if len(test_dataset) == 0:
        print("No test data found. Make sure to download and preprocess data first.")
        return
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Seizure samples: {(test_dataset.labels == 1).sum()}")
    print(f"Non-seizure samples: {(test_dataset.labels == 0).sum()}")
    
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = {}
    all_probs = {}
    
    # 1. Evaluate Transformer model
    print("\n" + "-"*40)
    print("Evaluating Transformer Model")
    print("-"*40)
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        encoder = create_encoder_from_config(config)
        classifier = SeizureClassifier(encoder)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        
        metrics, probs, labels = evaluate_model(
            classifier, test_loader, device=config.device
        )
        results['Transformer (ours)'] = metrics
        all_probs['Transformer (ours)'] = probs
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
    else:
        print(f"  Checkpoint not found: {checkpoint_path}")
        print("  Skipping transformer evaluation")
    
    # 2. Evaluate CNN Baseline
    print("\n" + "-"*40)
    print("Evaluating CNN Baseline")
    print("-"*40)
    
    cnn_baseline = BaselineCNN(
        n_channels=config.model.n_channels,
        n_samples=config.model.window_samples
    ).to(config.device)
    
    # Train CNN baseline quickly
    print("  Training CNN baseline...")
    train_dataset = EEGDataset(
        data_dir=config.data.data_dir,
        patients=config.data.train_patients,
        channels=config.data.common_channels,
        mode='finetune',
        balance_classes=True
    )
    
    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(cnn_baseline.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        cnn_baseline.train()
        for epoch in range(10):  # Quick training
            for batch in train_loader:
                x = batch['signal'].to(config.device)
                y = batch['label'].to(config.device)
                
                optimizer.zero_grad()
                output = cnn_baseline(x)
                loss = criterion(output['logits'], y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        metrics, probs, labels = evaluate_model(
            cnn_baseline, test_loader, device=config.device
        )
        results['CNN Baseline'] = metrics
        all_probs['CNN Baseline'] = probs
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
    else:
        print("  No training data for CNN baseline")
    
    # 3. Evaluate Random Forest Baseline
    print("\n" + "-"*40)
    print("Evaluating Random Forest Baseline")
    print("-"*40)
    
    if len(train_dataset) > 0:
        print("  Training Random Forest...")
        rf_baseline = RandomForestBaseline(n_estimators=100, max_depth=10)
        
        # Get numpy arrays for sklearn
        train_windows = train_dataset.windows
        train_labels = train_dataset.labels
        test_windows = test_dataset.windows
        test_labels = test_dataset.labels
        
        rf_baseline.fit(train_windows, train_labels)
        
        metrics, probs, labels = evaluate_baseline(
            rf_baseline, test_windows, test_labels
        )
        results['Random Forest'] = metrics
        all_probs['Random Forest'] = probs
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Generate plots
    if len(all_probs) > 0 and len(np.unique(labels)) > 1:
        print("\nGenerating plots...")
        plot_roc_curve(all_probs, labels, output_dir / "roc_curves.png")
        print(f"  ROC curves saved to {output_dir / 'roc_curves.png'}")
    
    # Generate report
    report = generate_report(results, output_dir)
    print(f"\nReport saved to {output_dir / 'evaluation_report.txt'}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
