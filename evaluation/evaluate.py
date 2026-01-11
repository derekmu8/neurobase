"""
Evaluation utilities for seizure detection models.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))


def evaluate_model(
    model,
    data_loader: DataLoader,
    device: str = 'mps',
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model with forward returning {'probs': ...}
        data_loader: DataLoader with {'signal': ..., 'label': ...}
        device: Device to run on
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model.to(device)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            x = batch['signal'].to(device)
            y = batch['label'].numpy()
            
            output = model(x)
            probs = output['probs'].cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(y)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'specificity': specificity_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auroc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        'n_samples': len(all_labels),
        'n_seizure': int(all_labels.sum()),
        'n_non_seizure': int(len(all_labels) - all_labels.sum())
    }
    
    return metrics, all_probs, all_labels


def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def evaluate_baseline(
    model,  # RandomForestBaseline
    windows: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate baseline model (sklearn-based).
    
    Args:
        model: Trained baseline model with predict_proba method
        windows: (n_samples, n_channels, n_samples)
        labels: (n_samples,)
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    probs = model.predict_proba(windows)
    preds = (probs > threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'specificity': specificity_score(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
        'auroc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        'n_samples': len(labels),
        'n_seizure': int(labels.sum()),
        'n_non_seizure': int(len(labels) - labels.sum())
    }
    
    return metrics, probs, labels


def plot_roc_curve(
    probs_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plot ROC curves for multiple models.
    
    Args:
        probs_dict: {'model_name': probs_array}
        labels: Ground truth labels
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return plt.gcf()


def plot_precision_recall_curve(
    probs_dict: Dict[str, np.ndarray],
    labels: np.ndarray,
    save_path: Optional[Path] = None
):
    """Plot precision-recall curves for multiple models."""
    plt.figure(figsize=(8, 6))
    
    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.plot(recall, precision, label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return plt.gcf()


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    # Header
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auroc']
    header = f"{'Model':<20}" + "".join(f"{m:<12}" for m in metrics)
    print(header)
    print("-"*70)
    
    # Rows
    for model_name, model_metrics in results.items():
        row = f"{model_name:<20}"
        for m in metrics:
            val = model_metrics.get(m, 0)
            row += f"{val:<12.4f}"
        print(row)
    
    print("="*70)


def generate_report(
    results: Dict[str, Dict[str, float]],
    save_dir: Path
) -> str:
    """
    Generate a text report of evaluation results.
    
    Args:
        results: {'model_name': metrics_dict}
        save_dir: Directory to save report
    
    Returns:
        Report string
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("="*70)
    report.append("NEUROBASE SEIZURE DETECTION - EVALUATION REPORT")
    report.append("="*70)
    report.append("")
    
    for model_name, metrics in results.items():
        report.append(f"\n{model_name}")
        report.append("-"*40)
        for k, v in metrics.items():
            if isinstance(v, float):
                report.append(f"  {k}: {v:.4f}")
            else:
                report.append(f"  {k}: {v}")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    
    # Save
    with open(save_dir / "evaluation_report.txt", 'w') as f:
        f.write(report_text)
    
    return report_text


if __name__ == "__main__":
    # Test with dummy data
    from models.encoder import EEGEncoder
    from models.classifier import SeizureClassifier
    
    # Create dummy model
    encoder = EEGEncoder()
    classifier = SeizureClassifier(encoder)
    
    # Dummy data
    class DummyDataset:
        def __init__(self, n_samples):
            self.n = n_samples
        
        def __len__(self):
            return self.n
        
        def __getitem__(self, idx):
            return {
                'signal': torch.randn(18, 1024),
                'label': np.float32(idx % 2)
            }
    
    from torch.utils.data import DataLoader
    loader = DataLoader(DummyDataset(50), batch_size=8)
    
    # Evaluate
    metrics, probs, labels = evaluate_model(classifier, loader, device='cpu')
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
