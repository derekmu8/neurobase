import numpy as np
from data_pipeline import load_and_preprocess_eeg
from feature_pipeline import calculate_power_envelope
from logic import find_hubs
from post_processing import debounce_hub_data

OUTPUT_FILENAME = 'hub_data.npy'
HUB_THRESHOLD = 0.87  # Optimal: ~5-6 spokes per frame on average

def main():
    """
    Main function to run the entire data processing pipeline.
    """
    print("--- NEURAL HUB FINDER: DATA PROCESSING ---")

    # Load and preprocess the EEG data
    print("Step 1/5: Loading and preprocessing data...")
    raw_data = load_and_preprocess_eeg()
    print("...Done.")

    # Calculate the power envelope for each channel
    print("Step 2/5: Calculating power envelope...")
    power_array = calculate_power_envelope(raw_data)
    print("...Done.")

    # Find hubs and spokes for each time point
    print("Step 3/5: Finding hubs and spokes...")
    # Note: To slice the array for the first 225 time samples for 15 seconds at 15 FPS
    hub_data = find_hubs(power_array[:, :225], threshold=HUB_THRESHOLD)
    print("...Done.")

    # Apply temporal debouncing to reduce flicker
    print("Step 4/5: Debouncing hub data...")
    smoothed_hub_data = debounce_hub_data(hub_data)
    print("...Done.")

    # Save the results to a file for the visualization script
    print(f"Step 5/5: Saving data to '{OUTPUT_FILENAME}'...")
    np.save(OUTPUT_FILENAME, smoothed_hub_data)
    print("...Done.")

    print("\n--- PROCESSING COMPLETE ---")
    print(f"Generated '{OUTPUT_FILENAME}' with {len(smoothed_hub_data)} animation frames.")

if __name__ == '__main__':
    main()
