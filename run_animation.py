import os
import sys
from pathlib import Path

os.environ['PYVISTA_OFF_SCREEN'] = 'true'
os.environ['MPLBACKEND'] = 'Agg' 

import numpy as np
import mne
import imageio.v2 as imageio 
import pyvista
from collections import deque

pyvista.OFF_SCREEN = True

# Set global plotting parameters to prevent interactive windows
import matplotlib
matplotlib.use('Agg') 

from dummy_data import generate_dummy_data
from visualizer import plot_brain_frame, project_sensors_to_surface, get_lobe_sensors

USE_DUMMY_DATA = False 

DATA_FILENAME = 'hub_data.npy'
OUTPUT_DIR = 'frames'
VIDEO_FILENAME = 'neural_hubs.mp4'
FPS = 15 

# Smoothing / fade configuration
HUB_SMOOTH_WINDOW = 12
HUB_WEIGHT_FLOOR = 0.35
HUB_EXTRA_OFFSET = 0.25
SPOKE_DECAY = 0.82
SPOKE_GROWTH = 0.55
SPOKE_MIN_STRENGTH = 0.1

class HubSpokeSmoother:
    """
    Applies a small amount of temporal smoothing so hubs/spokes fade in/out.
    """

    def __init__(
        self,
        hub_window=HUB_SMOOTH_WINDOW,
        decay=SPOKE_DECAY,
        growth=SPOKE_GROWTH,
        min_strength=SPOKE_MIN_STRENGTH,
        hub_weight_floor=HUB_WEIGHT_FLOOR,
        hub_extra_offset=HUB_EXTRA_OFFSET,
    ):
        self.hub_history = deque(maxlen=hub_window)
        self.decay = decay
        self.growth = growth
        self.min_strength = min_strength
        self.hub_weight_floor = hub_weight_floor
        self.hub_extra_offset = hub_extra_offset
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
        history_len = len(self.hub_history)
        weights = np.linspace(self.hub_weight_floor, 1.0, history_len)
        hub_scores = {}
        for hub_value, weight in zip(self.hub_history, weights):
            hub_scores[hub_value] = hub_scores.get(hub_value, 0.0) + weight
        smoothed_hub = max(hub_scores, key=hub_scores.get)
        total_weight = float(np.sum(weights)) if history_len else 1.0
        hub_strength = hub_scores[smoothed_hub] / total_weight

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

        return smoothed_hub, active_spokes, float(np.clip(hub_strength + self.hub_extra_offset, 0.0, 1.0))

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
        n_channels = None
        use_clinical = False
    else:
        print(f"Step 1/5: Loading real data from '{DATA_FILENAME}'...")
        loaded_array = np.load(DATA_FILENAME, allow_pickle=True)
        # Check if it's the new format with metadata
        if isinstance(loaded_array, np.ndarray) and loaded_array.dtype == object:
            loaded_item = loaded_array.item()
            if isinstance(loaded_item, dict) and 'hub_data' in loaded_item:
                # New format with metadata
                hub_data = loaded_item['hub_data']
                n_channels = loaded_item.get('n_channels')
                use_clinical = loaded_item.get('use_clinical_data', False)
                print(f"   Loaded data with {n_channels} channels (clinical: {use_clinical})")
            else:
                # Old format - just hub data
                hub_data = loaded_array.tolist()
                n_channels = None
                use_clinical = False
        else:
            # Old format
            hub_data = loaded_array.tolist()
            n_channels = None
            use_clinical = False
    print("...Done.")

    # 2. Get sensor locations - try to load from clinical data if available
    print("Step 2/5: Loading sensor location info...")
    mne.set_log_level('ERROR')
    
    # Check if we have clinical data file to get sensor positions from
    from run_processing import USE_CLINICAL_DATA, EDF_FILE_PATH
    # Use clinical data if the loaded data indicates it, or if configured
    should_use_clinical = use_clinical or (USE_CLINICAL_DATA and Path(EDF_FILE_PATH).exists())
    if should_use_clinical and Path(EDF_FILE_PATH).exists():
        print(f"Loading sensor info from clinical data: {EDF_FILE_PATH}")
        filepath_str = str(EDF_FILE_PATH)
        if filepath_str.endswith('.edf'):
            raw = mne.io.read_raw_edf(EDF_FILE_PATH, preload=False, verbose=False)
        elif filepath_str.endswith('.bdf'):
            raw = mne.io.read_raw_bdf(EDF_FILE_PATH, preload=False, verbose=False)
        else:
            raw = mne.io.read_raw(EDF_FILE_PATH, preload=False, verbose=False)
        raw.pick_types(meg=False, eeg=True, exclude="bads", verbose=False)
    else:
        # Fall back to MNE sample dataset
        print("Using MNE sample dataset for sensor locations")
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = (
            sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
        )
        raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
        raw.pick_types(meg=False, eeg=True, exclude="bads", verbose=False)
    
    # Get sensor positions
    try:
        sensor_locs = raw.get_montage().get_positions()['ch_pos']
        sensor_locs = np.array(list(sensor_locs.values()))
        
        # Check if sensor count matches the hub data
        if n_channels is not None and len(sensor_locs) != n_channels:
            print(f"Warning: Sensor count mismatch ({len(sensor_locs)} vs {n_channels}).")
            print("   Hub data has more channels than available sensor positions.")
            print("   Using sample dataset sensor positions (indices will be mapped).")
            raise ValueError("Sensor count mismatch")
            
    except Exception as e:
        print(f"Warning: Could not get sensor positions from clinical data: {e}")
        print("   Using sample dataset for sensor locations.")
        sample_data_folder = mne.datasets.sample.data_path()
        sample_data_raw_file = (
            sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
        )
        raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
        raw.pick_types(meg=False, eeg=True, exclude="bads", verbose=False)
        sensor_locs = raw.get_montage().get_positions()['ch_pos']
        sensor_locs = np.array(list(sensor_locs.values()))
        
        # If hub data has more channels, we need to map indices
        if n_channels is not None and n_channels > len(sensor_locs):
            print(f"   Mapping {n_channels} hub channels to {len(sensor_locs)} sensor positions.")
            print("   Using modulo mapping for visualization.")
            # We'll handle this in the plotting loop by using modulo
    
    info = raw.info
    
    # Get temporal lobe indices BEFORE projecting (for spatial detection)
    temporal_sensor_indices = get_lobe_sensors(info, lobe_name="temporal", sensor_locs=sensor_locs)
    sensor_locs = project_sensors_to_surface(sensor_locs)
    
    print(f"Identified {len(temporal_sensor_indices)} temporal lobe sensors: {temporal_sensor_indices}")
    print("...Done.")

    # 3. Prepare output directory
    print("Step 3/5: Preparing output directory...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("...Done.")

    # 4. Loop through frames and save each one as an image
    print(f"Step 4/5: Generating {len(hub_data)} animation frames...")
    smoother = HubSpokeSmoother()
    
    # Handle channel index mapping if needed
    n_sensor_locs = len(sensor_locs)
    map_indices = n_channels is not None and n_channels > n_sensor_locs
    
    for i, frame_data in enumerate(hub_data):
        hub = frame_data['hub']
        spokes = frame_data['spokes']
        
        # Map indices if hub data has more channels than sensor positions
        if map_indices:
            hub = hub % n_sensor_locs
            spokes = [s % n_sensor_locs for s in spokes]
        
        smoothed_hub, spoke_strengths, hub_strength = smoother.update(hub, spokes)
        spoke_indices = list(spoke_strengths.keys()) if spoke_strengths else spokes
        
        # Ensure indices are within bounds
        smoothed_hub = min(smoothed_hub, n_sensor_locs - 1)
        spoke_indices = [min(s, n_sensor_locs - 1) for s in spoke_indices]
        
        hub_coord_mm = sensor_locs[smoothed_hub] * 1000.0
        mean_spoke_strength = (
            float(np.mean(list(spoke_strengths.values())))
            if spoke_strengths
            else 0.0
        )
        max_spoke_strength = (
            float(np.max(list(spoke_strengths.values())))
            if spoke_strengths
            else 0.0
        )
        
        # Generate filename for this frame
        filename = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")

        frame_metadata = {
            'frame_index': i,
            'timestamp_seconds': i / FPS,
            'hub_label': smoothed_hub,
            'spoke_count': len(spoke_strengths),
            'fps': FPS,
            'hub_coord_mm': hub_coord_mm,
            'mean_spoke_strength': mean_spoke_strength,
            'max_spoke_strength': max_spoke_strength,
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
            temporal_sensor_indices=temporal_sensor_indices,
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