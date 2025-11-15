import os
import sys

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MPLBACKEND'] = 'Agg' 

import numpy as np
import mne
import imageio.v2 as imageio 
import pyvista

pyvista.OFF_SCREEN = True

# Set global plotting parameters to prevent interactive windows
import matplotlib
matplotlib.use('Agg') 

from dummy_data import generate_dummy_data
from visualizer import plot_brain_frame

USE_DUMMY_DATA = False 

DATA_FILENAME = 'hub_data.npy'
OUTPUT_DIR = 'frames'
VIDEO_FILENAME = 'neural_hubs.mp4'
FPS = 15 

def create_video_from_frames(frames_dir, video_filename, fps):
    """Stitches PNG frames into a video file."""
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    print(f"\nStitching {len(frame_files)} frames into video...")
    with imageio.get_writer(video_filename, fps=fps) as writer:
        for filename in frame_files:
            image = imageio.imread(os.path.join(frames_dir, filename))
            writer.append_data(image)
    print(f"Video saved as '{video_filename}'")

def main():
    """Main function to run the entire animation pipeline."""
    print("--- NEURAL HUB FINDER: ANIMATION ---")

    # 1. Load the data
    if USE_DUMMY_DATA:
        print("Step 1/5: Generating dummy data...")
        hub_data = generate_dummy_data()
    else:
        print(f"Step 1/5: Loading real data from '{DATA_FILENAME}'...")
        loaded_array = np.load(DATA_FILENAME, allow_pickle=True)
        hub_data = loaded_array.tolist() 
    print("...Done.")

    # 2. Get sensor locations from the MNE sample dataset
    print("Step 2/5: Loading sensor location info...")
    sample_data_folder = mne.datasets.sample.data_path()
    sample_data_raw_file = (
        sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    )
    # Suppress verbose output from MNE
    mne.set_log_level('ERROR')
    raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
    raw.pick_types(meg=False, eeg=True, exclude="bads", verbose=False)
    
    sensor_locs = raw.get_montage().get_positions()['ch_pos']
    
    sensor_locs = np.array(list(sensor_locs.values()))
    info = raw.info
    print("...Done.")

    # 3. Prepare output directory
    print("Step 3/5: Preparing output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("...Done.")

    # 4. Loop through frames and save each one as an image
    print(f"Step 4/5: Generating {len(hub_data)} animation frames...")
    for i, frame_data in enumerate(hub_data):
        hub = frame_data['hub']
        spokes = frame_data['spokes']
        
        # Generate filename for this frame
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
        
        # Plot and save the frame (plotter is created and closed inside the function)
        plot_brain_frame(info, hub, spokes, sensor_locs, filename)
        
        # Print progress
        print(f"  ...saved {filename}", end='\r')
    print("\n...Done.")

    # 5. Stitch the frames together into a video
    print("Step 5/5: Creating final video...")
    create_video_from_frames(OUTPUT_DIR, VIDEO_FILENAME, FPS)
    print("...Done.")
    
    print("\n--- ANIMATION COMPLETE ---")

if __name__ == '__main__':
    main()