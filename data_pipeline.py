import mne

L_FREQ = 8.0  # Lower bound of the alpha band (in Hz)
H_FREQ = 13.0 # Upper bound of the alpha band (in Hz)

def load_and_preprocess_eeg():
    """
    Loads the MNE sample dataset, extracts the EEG data, and applies a
    band-pass filter to isolate the alpha frequency band.

    Returns:
        mne.io.Raw: The preprocessed MNE Raw object containing EEG data.
    """
    # Load the MNE sample dataset
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    # Load the data into memory
    raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True, verbose=False)

    # Select only the EEG channels
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    # Apply a band-pass filter to isolate alpha waves
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
