import numpy as np

NUM_CHANNELS = 59  # The MNE sample dataset has 59 EEG channels
NUM_FRAMES = 100   # Let's generate 100 frames for a short test animation

def generate_dummy_data():
    """
    Generates a list of fake hub/spoke dictionaries for testing.
    This data matches the format of the 'data_contract_example.py'.

    Returns:
        list: A list of dummy data dictionaries.
    """
    dummy_hub_data = []
    for _ in range(NUM_FRAMES):
        # Pick a random channel to be the hub
        hub_idx = np.random.randint(0, NUM_CHANNELS)
        
        # Pick a random number of spokes (2-8)
        num_spokes = np.random.randint(2, 9)
        
        # Pick random channels to be spokes
        spoke_indices = np.random.choice(
            NUM_CHANNELS, size=num_spokes, replace=False
        ).tolist()
        
        # Ensure the hub is not listed as a spoke
        if hub_idx in spoke_indices:
            spoke_indices.remove(hub_idx)
            
        dummy_hub_data.append({
            'hub': hub_idx,
            'spokes': spoke_indices
        })
        
    return dummy_hub_data

if __name__ == '__main__':
    # A simple test to see what the data looks like
    test_data = generate_dummy_data()
    print("--- Generated Dummy Data Sample ---")
    for i, frame in enumerate(test_data[:3]):
        print(f"Frame {i}: Hub={frame['hub']}, Spokes={frame['spokes']}")