"""
Models module for NeuroBase
"""
from .encoder import EEGEncoder, create_encoder_from_config
from .pretrain import MaskedPatchPredictor, ContrastivePretrainer, create_pretrainer
from .classifier import SeizureClassifier, BaselineCNN, create_classifier
from .baseline import RandomForestBaseline, extract_features

__all__ = [
    'EEGEncoder',
    'create_encoder_from_config',
    'MaskedPatchPredictor',
    'ContrastivePretrainer',
    'create_pretrainer',
    'SeizureClassifier',
    'BaselineCNN',
    'create_classifier',
    'RandomForestBaseline',
    'extract_features'
]
