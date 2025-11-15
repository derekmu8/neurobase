import numpy as np
from scipy.signal import hilbert

def calculate_power_envelope(raw_data):
    """
    Calculates the power envelope of the EEG signals using the Hilbert transform.

    Args:
        raw_data (mne.io.Raw): The preprocessed MNE Raw object.

    Returns:
        np.ndarray: A 2D array of shape (n_channels, n_times) containing
                    the power of the signal for each channel over time.
    """
    # Get the EEG data as a NumPy array
    eeg_data = raw_data.get_data()

    # Calculate the analytic signal using the Hilbert transform
    # The absolute value of this complex signal gives us the instantaneous amplitude (envelope).
    analytic_signal = hilbert(eeg_data, axis=1)
    power_envelope = np.abs(analytic_signal)

    return power_envelope

if __name__ == '__main__':
    pass
