import numpy as np

FILENAME = 'hub_data.npy'

def inspect_npy_file(filename):
    """
    Loads and prints information about the hub_data.npy file.
    """
    try:
        loaded_array = np.load(filename, allow_pickle=True)
        data = loaded_array.tolist()

        print(f"--- Inspection of '{filename}' ---")
        print(f"File loaded successfully.")
        print(f"Type of loaded data: {type(data)}")

        if isinstance(data, list):
            print(f"Total number of animation frames: {len(data)}")
            
            if len(data) > 0:
                print("\n--- Example Frames ---")
                
                # Print the first 3 frames
                for i, frame in enumerate(data[:3]):
                    print(f"\nFrame {i}:")
                    print(f"  Type: {type(frame)}")
                    if isinstance(frame, dict):
                        hub = frame.get('hub', 'N/A')
                        spokes = frame.get('spokes', [])
                        print(f"  Hub Index: {hub}")
                        print(f"  Spoke Indices: {spokes}")
                        print(f"  Number of Spokes: {len(spokes)}")
                    else:
                        print(f"  Unexpected data in frame: {frame}")
            else:
                print("The file contains an empty list.")
        else:
            print(f"Loaded data is not a list as expected. Data: {data}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    inspect_npy_file(FILENAME)
