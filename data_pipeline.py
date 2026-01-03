import mne
import tempfile
import urllib.request

L_FREQ = 8.0  # Lower bound of the alpha band (in Hz)
H_FREQ = 13.0 # Upper bound of the alpha band (in Hz)

# URL to a clinical EEG dataset from PhysioNet (CHB-MIT, seizure data)
DATA_URL = 'https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf'

def load_and_preprocess_eeg():
    """
    Loads clinical EEG data from OpenNeuro ds003029 (seizure data), extracts the EEG data, and applies a
    band-pass filter to isolate the alpha frequency band.

    Returns:
        mne.io.Raw: The preprocessed MNE Raw object containing EEG data.
    """
    # Suppress verbose output from MNE
    mne.set_log_level('ERROR')
    
    # Download the EDF file
    print(f"Downloading clinical EEG data from {DATA_URL}...")
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp_file:
        tmp_filename = tmp_file.name
    urllib.request.urlretrieve(DATA_URL, tmp_filename)
    print("Download complete.")
    
    # Load the data
    raw = mne.io.read_raw_edf(tmp_filename, preload=True, verbose=False)

    # Select only the EEG channels
    raw.pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads", verbose=False)

    # Apply a band-pass filter to isolate alpha waves
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", verbose=False)

    # Clean up temp file
    import os
    os.unlink(tmp_filename)

    return raw

if __name__ == '__main__':
    """
    Run to test: python data_pipeline.py
    """
    print("Running data pipeline test...")
    preprocessed_data = load_and_preprocess_eeg()
    print("Data loaded and preprocessed successfully!")
    print(preprocessed_data.info)
