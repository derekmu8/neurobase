import os
import sys

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MPLBACKEND'] = 'Agg' 

import numpy as np
import mne
import imageio.v2 as imageio 
import pyvista
from collections import Counter, deque

pyvista.OFF_SCREEN = True

# Set global plotting parameters to prevent interactive windows
import matplotlib
matplotlib.use('Agg') 

from dummy_data import generate_dummy_data
from visualizer import plot_brain_frame, project_sensors_to_surface

USE_DUMMY_DATA = False 

DATA_FILENAME = 'hub_data.npy'
OUTPUT_DIR = 'frames'
VIDEO_FILENAME = 'neural_hubs.mp4'
FPS = 15 

class HubSpokeSmoother:
    """
    Applies a small amount of temporal smoothing so hubs/spokes fade in/out.
    """

    def __init__(self, hub_window=5, decay=0.8, growth=0.5, min_strength=0.12):
        self.hub_history = deque(maxlen=hub_window)
        self.decay = decay
        self.growth = growth
        self.min_strength = min_strength
        self.spoke_strengths = {}

    def _decay_spokes(self):
        to_delete = []
        for idx, strength in self.spoke_strengths.items():
            new_strength = strength * self.decay
            if new_strength < self.min_strength / 2:
                to_delete.append(idx)
            else:
                self.spoke_strengths[idx] = new_strength
        for idx in to_delete:
            del self.spoke_strengths[idx]

    def update(self, hub_idx, spokes):
        self.hub_history.append(hub_idx)
        hub_counts = Counter(self.hub_history)
        smoothed_hub = hub_counts.most_common(1)[0][0]
        hub_strength = hub_counts[smoothed_hub] / len(self.hub_history)

        self._decay_spokes()
        for spoke in spokes:
            self.spoke_strengths[spoke] = min(
                1.0, self.spoke_strengths.get(spoke, 0.0) + self.growth
            )

        active_spokes = {
            idx: strength
            for idx, strength in self.spoke_strengths.items()
            if strength >= self.min_strength
        }
        self.spoke_strengths = active_spokes.copy()

        return smoothed_hub, active_spokes, float(np.clip(hub_strength + 0.25, 0.0, 1.0))

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
    sensor_locs = project_sensors_to_surface(sensor_locs)
    info = raw.info
    print("...Done.")

    # 3. Prepare output directory
    print("Step 3/5: Preparing output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("...Done.")

    # 4. Loop through frames and save each one as an image
    print(f"Step 4/5: Generating {len(hub_data)} animation frames...")
    smoother = HubSpokeSmoother()
    for i, frame_data in enumerate(hub_data):
        hub = frame_data['hub']
        spokes = frame_data['spokes']
        smoothed_hub, spoke_strengths, hub_strength = smoother.update(hub, spokes)
        spoke_indices = list(spoke_strengths.keys()) if spoke_strengths else spokes
        
        # Generate filename for this frame
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")

        frame_metadata = {
            'frame_index': i,
            'timestamp_seconds': i / FPS,
            'hub_label': smoothed_hub,
            'spoke_count': len(spoke_strengths),
            'fps': FPS,
        }
        
        # Plot and save the frame (plotter is created and closed inside the function)
        plot_brain_frame(
            info,
            smoothed_hub,
            spoke_indices,
            sensor_locs,
            filename,
            frame_metadata=frame_metadata,
            spoke_strengths=spoke_strengths if spoke_strengths else None,
            hub_strength=hub_strength,
        )
        
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