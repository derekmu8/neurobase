import numpy as np
from data_pipeline import load_and_preprocess_eeg, load_clinical_seizure_data
from feature_pipeline import calculate_power_envelope
from logic import find_hubs

OUTPUT_FILENAME = 'hub_data.npy'
HUB_THRESHOLD = 0.87  # Optimal: ~5-6 spokes per frame on average

# Clinical data configuration
USE_CLINICAL_DATA = True  # Set to True to use clinical seizure data
EDF_FILE_PATH = 'data/sample_clinical.edf'  # Update with your file path
SEIZURE_ONSET_TIME = 45  # seconds - Set to None for auto-detection, or specify manually
CLIP_DURATION = 30  # seconds of pre-seizure data to extract

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("--- NEURAL HUB FINDER: DATA PROCESSING ---")

    # Load and preprocess the EEG data
    print("Step 1/4: Loading and preprocessing data...")
    if USE_CLINICAL_DATA:
        print(f"Using clinical data from: {EDF_FILE_PATH}")
        raw_data = load_clinical_seizure_data(
            EDF_FILE_PATH, 
            seizure_onset_time=SEIZURE_ONSET_TIME, 
            duration_seconds=CLIP_DURATION,
            auto_detect_seizure=(SEIZURE_ONSET_TIME is None)
        )
    else:
        print("Using MNE sample dataset")
        raw_data = load_and_preprocess_eeg()
    print("...Done.")

    # Calculate the power envelope for each channel
    print("Step 2/4: Calculating power envelope...")
    power_array = calculate_power_envelope(raw_data)
    print("...Done.")

    # Find hubs and spokes for each time point
    print("Step 3/4: Finding hubs and spokes...")
    # Note: To slice the array for the first 3000 time samples: power_array[:, :3000]
    hub_data = find_hubs(power_array[:, :50], threshold=HUB_THRESHOLD)
    print("...Done.")

    # Save the results to a file for the visualization script
    print(f"Step 4/4: Saving data to '{OUTPUT_FILENAME}'...")
    # Save hub data along with metadata about number of channels
    save_data = {
        'hub_data': hub_data,
        'n_channels': len(raw_data.ch_names),
        'use_clinical_data': USE_CLINICAL_DATA,
        'edf_file_path': EDF_FILE_PATH if USE_CLINICAL_DATA else None
    }
    np.save(OUTPUT_FILENAME, save_data)
    print("...Done.")

    print("\n--- PROCESSING COMPLETE ---")
    print(f"Generated '{OUTPUT_FILENAME}' with {len(hub_data)} animation frames.")

if __name__ == '__main__':
    main()
