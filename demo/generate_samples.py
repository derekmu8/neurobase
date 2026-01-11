"""
Pre-generated sample data for demo
"""
import numpy as np
from pathlib import Path

def generate_demo_samples():
    """Generate synthetic EEG samples for demo"""
    samples_dir = Path(__file__).parent / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    np.random.seed(42)
    t = np.linspace(0, 4, 1024)
    
    # Normal EEG-like signal
    normal_signal = np.zeros((18, 1024))
    for ch in range(18):
        # Mix of alpha (10Hz) and beta (20Hz) rhythms
        normal_signal[ch] = (
            0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi) +
            0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi) +
            0.2 * np.random.randn(1024)
        )
    
    np.save(samples_dir / "normal_eeg.npy", {
        'signal': normal_signal.astype(np.float32),
        'label': 0,
        'description': 'Normal EEG activity'
    })
    
    # Seizure-like signal (high amplitude, rhythmic spikes)
    seizure_signal = np.zeros((18, 1024))
    for ch in range(18):
        # Seizure pattern: high amplitude rhythmic activity
        base = 2.0 * np.sin(2 * np.pi * 3 * t)  # 3Hz spike-wave
        spikes = np.zeros(1024)
        spike_times = np.arange(0, 1024, 85)  # Regular spikes
        for st in spike_times:
            if st + 20 < 1024:
                spikes[st:st+20] = 3.0 * np.exp(-np.linspace(0, 3, 20))
        seizure_signal[ch] = base + spikes + 0.3 * np.random.randn(1024)
    
    np.save(samples_dir / "seizure_eeg.npy", {
        'signal': seizure_signal.astype(np.float32),
        'label': 1,
        'description': 'Seizure activity (3Hz spike-wave pattern)'
    })
    
    print(f"Demo samples saved to {samples_dir}")


if __name__ == "__main__":
    generate_demo_samples()
