"""
Baseline models for comparison.

1. Random Forest on handcrafted features
2. Simple CNN (already in classifier.py)
"""
import numpy as np
from typing import List, Tuple
from scipy import stats
from scipy.signal import welch


def extract_features(windows: np.ndarray, sfreq: int = 256) -> np.ndarray:
    """
    Extract handcrafted features from EEG windows.
    
    Features per channel:
    - Statistical: mean, std, skewness, kurtosis, min, max
    - Frequency: band powers (delta, theta, alpha, beta, gamma)
    - Line length
    
    Args:
        windows: (n_windows, n_channels, n_samples)
        sfreq: sampling frequency
    
    Returns:
        features: (n_windows, n_features)
    """
    n_windows, n_channels, n_samples = windows.shape
    
    features_list = []
    
    for w in range(n_windows):
        window_features = []
        
        for ch in range(n_channels):
            signal = windows[w, ch]
            
            # Statistical features
            window_features.extend([
                np.mean(signal),
                np.std(signal),
                stats.skew(signal),
                stats.kurtosis(signal),
                np.min(signal),
                np.max(signal),
                np.ptp(signal),  # peak-to-peak
            ])
            
            # Line length (sum of absolute differences)
            line_length = np.sum(np.abs(np.diff(signal)))
            window_features.append(line_length)
            
            # Frequency band powers
            freqs, psd = welch(signal, sfreq, nperseg=min(256, n_samples))
            
            # Band definitions (Hz)
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }
            
            for band_name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.trapz(psd[idx], freqs[idx]) if idx.any() else 0
                window_features.append(band_power)
        
        features_list.append(window_features)
    
    return np.array(features_list, dtype=np.float32)


class RandomForestBaseline:
    """
    Random Forest classifier on handcrafted features.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.sfreq = 256
    
    def fit(self, windows: np.ndarray, labels: np.ndarray):
        """
        Train the model.
        
        Args:
            windows: (n_samples, n_channels, n_samples)
            labels: (n_samples,)
        """
        print("Extracting features...")
        features = extract_features(windows, self.sfreq)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print("Fitting scaler and model...")
        features = self.scaler.fit_transform(features)
        self.model.fit(features, labels)
        
        print(f"Trained on {len(labels)} samples")
    
    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Predict labels"""
        features = extract_features(windows, self.sfreq)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        features = extract_features(windows, self.sfreq)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self.scaler.transform(features)
        return self.model.predict_proba(features)[:, 1]
    
    def evaluate(
        self, 
        windows: np.ndarray, 
        labels: np.ndarray
    ) -> dict:
        """Evaluate model performance"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        preds = self.predict(windows)
        probs = self.predict_proba(windows)
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'auroc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        }


if __name__ == "__main__":
    # Test feature extraction
    windows = np.random.randn(10, 18, 1024).astype(np.float32)
    features = extract_features(windows)
    print(f"Features shape: {features.shape}")
    print(f"Features per window: {features.shape[1]}")
    
    # Test model
    labels = np.random.randint(0, 2, 10)
    model = RandomForestBaseline()
    model.fit(windows, labels)
    
    preds = model.predict(windows)
    print(f"Predictions: {preds}")
