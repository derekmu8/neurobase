"""
Evaluation module for NeuroBase
"""
from .evaluate import (
    evaluate_model,
    evaluate_baseline,
    plot_roc_curve,
    plot_precision_recall_curve,
    print_comparison_table,
    generate_report
)

__all__ = [
    'evaluate_model',
    'evaluate_baseline',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'print_comparison_table',
    'generate_report'
]
