import numpy as np
import pandas as pd
from scipy.signal import hilbert

def calculate_power_envelope(raw_data, window_ms=100):
    """
    Calculates the power envelope of the EEG signals using the Hilbert transform,
    with temporal smoothing to reduce the impact of noisy channels and reveal
    dynamic changes over time.

    Args:
        raw_data (mne.io.Raw): The preprocessed MNE Raw object.
        window_ms (float): The size of the rolling window in milliseconds for
                          temporal smoothing (default: 100ms).

    Returns:
        np.ndarray: A 2D array of shape (n_channels, n_times) containing
                    the smoothed power of the signal for each channel over time.
    """
    # Get the EEG data as a NumPy array
    eeg_data = raw_data.get_data()

    # Calculate the analytic signal using the Hilbert transform
    # The absolute value of this complex signal gives us the instantaneous amplitude (envelope).
    analytic_signal = hilbert(eeg_data, axis=1)
    power_envelope = np.abs(analytic_signal)

    # Apply temporal smoothing using a rolling window average
    # This reduces the impact of noisy channels and makes hub dynamics visible
    
    # Get the sampling frequency to convert window_ms to samples
    sfreq = raw_data.info['sfreq']
    window_samples = int(window_ms * sfreq / 1000)
    
    # Use pandas to apply rolling window smoothing to each channel
    smoothed_power = np.zeros_like(power_envelope)
    for ch_idx in range(power_envelope.shape[0]):
        # Create a pandas Series for this channel
        channel_series = pd.Series(power_envelope[ch_idx, :])
        
        # Apply rolling mean with centered window and min_periods to handle edges
        smoothed_series = channel_series.rolling(
            window=window_samples,
            center=True,
            min_periods=1
        ).mean()
        
        # Store the smoothed data back into the array
        smoothed_power[ch_idx, :] = smoothed_series.values

    return smoothed_power

if __name__ == '__main__':
    pass
