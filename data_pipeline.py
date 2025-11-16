import mne
from pathlib import Path

L_FREQ = 8.0  # Lower bound of the alpha band (in Hz)
H_FREQ = 13.0 # Upper bound of the alpha band (in Hz)

def load_and_preprocess_eeg():
    """
    Loads the MNE sample dataset, extracts the EEG data, and applies a
    band-pass filter to isolate the alpha frequency band.

    Returns:
        mne.io.Raw: The preprocessed MNE Raw object containing EEG data.
    """
    # Suppress verbose output from MNE
    mne.set_log_level('ERROR')
    
    # Load the MNE sample dataset
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    # Load the data into memory
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

    # Select only the EEG channels
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads", verbose=False)

    # Apply a band-pass filter to isolate alpha waves
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)

    return raw

def load_clinical_seizure_data(filepath, seizure_onset_time, duration_seconds=30):
    """
    Loads a clinical EDF file and extracts a pre-seizure segment for analysis.
    
    Args:
        filepath (str | Path): Path to the EDF file containing EEG data.
        seizure_onset_time (float): Time in seconds when the seizure occurs.
        duration_seconds (float): Duration of the pre-seizure segment to extract (default: 30 seconds).
    
    Returns:
        mne.io.Raw: The preprocessed MNE Raw object containing the pre-seizure EEG data.
    """
    # Suppress verbose output from MNE
    mne.set_log_level('ERROR')
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EDF file not found: {filepath}")
    
    # Load the EDF file
    print(f"Loading EDF file: {filepath}")
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    # Set a standard montage for 3D sensor positions
    # This is important for visualization
    try:
        raw.set_montage('standard_1020', on_missing='warn', verbose=False)
    except Exception as e:
        print(f"Warning: Could not set standard_1020 montage: {e}")
        print("Attempting to use existing montage...")
    
    # Select only EEG channels (exclude EOG, ECG, etc.)
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, ecg=False, exclude="bads", verbose=False)
    
    # Calculate the time window for the pre-seizure segment
    start_time = max(0, seizure_onset_time - duration_seconds)
    end_time = seizure_onset_time
    
    # Crop to the pre-seizure window
    print(f"Extracting pre-seizure segment: {start_time:.1f}s to {end_time:.1f}s")
    raw.crop(tmin=start_time, tmax=end_time, verbose=False)
    
    # Apply the same band-pass filter as the original function
    # For seizure analysis, we might want to look at different frequency bands
    # but keeping alpha for consistency with the original pipeline
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)
    
    return raw

if __name__ == '__main__':
    """
    Run to test: python data_pipeline.py
    """
    print("Running data pipeline test...")
    preprocessed_data = load_and_preprocess_eeg()
    print("Data loaded and preprocessed successfully!")
    print(preprocessed_data.info)
