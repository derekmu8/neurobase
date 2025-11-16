import numpy as np
from data_pipeline import load_and_preprocess_eeg
from feature_pipeline import calculate_power_envelope
from logic import find_hubs

OUTPUT_FILENAME = 'hub_data.npy'
HUB_THRESHOLD = 0.87  # Optimal: ~5-6 spokes per frame on average

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("--- NEURAL HUB FINDER: DATA PROCESSING ---")

    # Load and preprocess the EEG data
    print("Step 1/4: Loading and preprocessing data...")
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
    np.save(OUTPUT_FILENAME, hub_data)
    print("...Done.")

    print("\n--- PROCESSING COMPLETE ---")
    print(f"Generated '{OUTPUT_FILENAME}' with {len(hub_data)} animation frames.")

if __name__ == '__main__':
    main()
