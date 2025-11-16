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

def find_seizure_onset_time(raw, seizure_keywords=None):
    """
    Attempts to automatically find seizure onset time from annotations or events.
    
    Args:
        raw (mne.io.Raw): The loaded raw data object.
        seizure_keywords (list | None): Keywords to search for in annotations.
            Default: ['seizure', 'ictal', 'onset', 'sz']
    
    Returns:
        float | None: Seizure onset time in seconds, or None if not found.
    """
    if seizure_keywords is None:
        seizure_keywords = ['seizure', 'ictal', 'onset', 'sz', 'epilepsy']
    
    # Check annotations
    if raw.annotations is not None and len(raw.annotations) > 0:
        for desc in raw.annotations.description:
            desc_lower = desc.lower()
            if any(keyword in desc_lower for keyword in seizure_keywords):
                # Return the onset time of the first matching annotation
                idx = list(raw.annotations.description).index(desc)
                onset_time = raw.annotations.onset[idx]
                print(f"Found seizure annotation: '{desc}' at {onset_time:.2f}s")
                return float(onset_time)
    
    # Check events if available
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if events is not None and len(events) > 0:
            for event_name, event_code in event_id.items():
                event_name_lower = event_name.lower()
                if any(keyword in event_name_lower for keyword in seizure_keywords):
                    # Find first event with this code
                    matching_events = events[events[:, 2] == event_code]
                    if len(matching_events) > 0:
                        onset_time = matching_events[0, 0] / raw.info['sfreq']
                        print(f"Found seizure event: '{event_name}' at {onset_time:.2f}s")
                        return float(onset_time)
    except Exception:
        pass
    
    return None

def load_clinical_seizure_data(filepath, seizure_onset_time=None, duration_seconds=30, auto_detect_seizure=True):
    """
    Loads a clinical EDF file and extracts a pre-seizure segment for analysis.
    
    Args:
        filepath (str | Path): Path to the EDF file containing EEG data.
        seizure_onset_time (float | None): Time in seconds when the seizure occurs.
            If None and auto_detect_seizure=True, will attempt to find it automatically.
        duration_seconds (float): Duration of the pre-seizure segment to extract (default: 30 seconds).
        auto_detect_seizure (bool): If True, attempt to automatically detect seizure onset from annotations.
    
    Returns:
        mne.io.Raw: The preprocessed MNE Raw object containing the pre-seizure EEG data.
    """
    # Suppress verbose output from MNE
    mne.set_log_level('ERROR')
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EDF file not found: {filepath}")
    
    # Load the EEG file (supports EDF, BDF, and other MNE-supported formats)
    print(f"Loading EEG file: {filepath}")
    filepath_str = str(filepath)
    
    if filepath_str.endswith('.edf'):
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    elif filepath_str.endswith('.bdf'):
        raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
    else:
        # Try to auto-detect format
        raw = mne.io.read_raw(filepath, preload=True, verbose=False)
    
    # Auto-detect seizure onset if not provided
    if seizure_onset_time is None and auto_detect_seizure:
        print("Attempting to auto-detect seizure onset time...")
        detected_time = find_seizure_onset_time(raw)
        if detected_time is not None:
            seizure_onset_time = detected_time
        else:
            print("Warning: Could not auto-detect seizure onset. Using middle of recording.")
            seizure_onset_time = raw.times[-1] / 2
    
    if seizure_onset_time is None:
        raise ValueError("seizure_onset_time must be provided or auto-detected")
    
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
