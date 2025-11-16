import numpy as np
import mne
import pyvista
import time
from collections import deque

from dummy_data import generate_dummy_data
from visualizer import project_sensors_to_surface, _load_brain_surface

USE_DUMMY_DATA = False 

DATA_FILENAME = 'hub_data.npy'
# Interactive animations can feel too fast at video-level FPS; slow to 2 FPS
FPS = 2 

# Smoothing / fade configuration
# Make transitions smoother by extending the hub window and slowing spoke decay
HUB_SMOOTH_WINDOW = 18
HUB_WEIGHT_FLOOR = 0.35
HUB_EXTRA_OFFSET = 0.25
SPOKE_DECAY = 0.93
SPOKE_GROWTH = 0.45
SPOKE_MIN_STRENGTH = 0.05

# Visualization constants from visualizer.py
SHOW_BRAIN = True
SENSOR_SURFACE_OFFSET = 0.003
CAMERA_OFFSET = np.array([0.22, -0.25, 0.15])
CAMERA_UP = (0, 0, 1)

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

# Global variables for animation callback
current_frame_index = 0
hub_data = None
sensor_locs = None
smoother = None
plotter = None
dynamic_actors = []  # Track dynamic actors (hubs, spokes, lines) for removal
last_update_time = 0.0
frame_interval = 1.0 / FPS  # Time between frames in seconds

def animation_callback(plotter_obj=None, force=False):
    """
    Callback function that updates the visualization for each frame.
    Removes previous dynamic actors and adds new ones.
    Uses time-based throttling to control frame rate.
    """
    global current_frame_index, hub_data, sensor_locs, smoother, dynamic_actors, plotter
    global last_update_time, frame_interval
    
    # Use plotter_obj if provided (from render callback), otherwise use global
    active_plotter = plotter_obj if plotter_obj is not None else plotter
    
    if hub_data is None or len(hub_data) == 0 or active_plotter is None:
        return
    
    # Throttle updates based on frame rate unless forced (e.g., initial draw)
    current_time = time.time()
    if not force and (current_time - last_update_time) < frame_interval:
        return
    last_update_time = current_time
    
    # Get frame data
    frame_data = hub_data[current_frame_index]
    hub = frame_data['hub']
    spokes = frame_data['spokes']
    
    # Apply smoothing
    smoothed_hub, spoke_strengths, hub_strength = smoother.update(hub, spokes)
    spoke_indices = list(spoke_strengths.keys()) if spoke_strengths else spokes
    
    # Remove previous dynamic actors
    for actor in dynamic_actors:
        try:
            active_plotter.remove_actor(actor)
        except:
            pass
    dynamic_actors.clear()
    
    # Add new spokes
    if spoke_indices:
        spoke_locs = sensor_locs[spoke_indices]
        for idx, (spoke_idx, spoke_loc) in enumerate(zip(spoke_indices, spoke_locs)):
            strength = float(np.clip(spoke_strengths.get(spoke_idx, 0.0), 0.0, 1.0))
            radius = 0.006 + 0.004 * strength
            opacity = 0.35 + 0.45 * strength
            sphere = pyvista.Sphere(radius=radius, center=spoke_loc, phi_resolution=28, theta_resolution=28)
            actor = active_plotter.add_mesh(
                sphere,
                color='yellow',
                smooth_shading=True,
                opacity=opacity,
                specular=0.55,
                specular_power=35,
            )
            dynamic_actors.append(actor)
    
    # Add new hub
    hub_loc = sensor_locs[smoothed_hub]
    hub_strength = float(np.clip(hub_strength, 0.0, 1.0))
    hub_radius = 0.012 + 0.010 * hub_strength
    hub_opacity = 0.6 + 0.35 * hub_strength
    hub_sphere = pyvista.Sphere(radius=hub_radius, center=hub_loc, phi_resolution=32, theta_resolution=32)
    actor = active_plotter.add_mesh(
        hub_sphere,
        color='red',
        smooth_shading=True,
        opacity=hub_opacity,
        specular=0.6,
        specular_power=35,
    )
    dynamic_actors.append(actor)
    
    # Add new connection lines
    for idx, spoke_idx in enumerate(spoke_indices):
        spoke_loc = sensor_locs[spoke_idx]
        line = pyvista.Line(hub_loc, spoke_loc)
        raw_strength = spoke_strengths.get(spoke_idx, 0.0) if spoke_strengths else 1.0
        strength = float(np.clip(raw_strength, 0.0, 1.0))
        line_width = 2 + 4 * strength
        opacity = 0.35 + 0.55 * strength
        actor = active_plotter.add_mesh(line, color='cyan', line_width=line_width, opacity=opacity)
        dynamic_actors.append(actor)
    
    # Update frame index (loop)
    current_frame_index = (current_frame_index + 1) % len(hub_data)

def main():
    """Main function to run the interactive demo."""
    global current_frame_index, hub_data, sensor_locs, smoother, plotter
    
    print("--- NEURAL HUB FINDER: INTERACTIVE DEMO ---")

    # 1. Load the data
    if USE_DUMMY_DATA:
        print("Step 1/4: Generating dummy data...")
        hub_data = generate_dummy_data()
    else:
        print(f"Step 1/4: Loading real data from '{DATA_FILENAME}'...")
        loaded_array = np.load(DATA_FILENAME, allow_pickle=True)
        hub_data = loaded_array.tolist() 
    print("...Done.")

    # 2. Get sensor locations from the MNE sample dataset
    print("Step 2/4: Loading sensor location info...")
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

    # 3. Initialize smoother
    print("Step 3/4: Initializing smoother...")
    smoother = HubSpokeSmoother()
    print("...Done.")

    # 4. Initialize PyVista plotter for on-screen rendering
    print("Step 4/4: Initializing interactive plotter...")
    plotter = pyvista.Plotter(window_size=[1024, 768])
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

    # Set up the static brain mesh
    brain_mesh = None
    brain_center = np.mean(sensor_locs, axis=0)
    if SHOW_BRAIN:
        brain_mesh, coords, _ = _load_brain_surface()
        if coords is not None:
            brain_center = np.mean(coords, axis=0)
        if brain_mesh is None:
            fallback_radius = 1.05 * np.max(np.linalg.norm(sensor_locs - brain_center, axis=1))
            brain_mesh = pyvista.Sphere(
                radius=fallback_radius,
                center=brain_center,
                theta_resolution=80,
                phi_resolution=80,
            )
        
        # Add brain mesh as static element
        if brain_mesh is not None:
            brain_display = brain_mesh.copy(deep=True) if hasattr(brain_mesh, "copy") else brain_mesh
            plotter.add_mesh(
                brain_display,
                color='#d0d0d0',
                opacity=0.4,
                smooth_shading=False,
                ambient=0.35,
                specular=0.05,
                name="scalp"
            )
    
    # Set background color
    plotter.set_background('black')
    
    # Set the camera view
    focus_point = brain_center if np.any(brain_center) else np.array([0.0, 0.0, 0.0])
    camera_pos = focus_point + CAMERA_OFFSET
    plotter.camera_position = [
        tuple(camera_pos.tolist()),
        tuple(focus_point.tolist()),
        CAMERA_UP,
    ]
    plotter.camera.zoom(0.85)
    plotter.reset_camera_clipping_range()
    
    print("...Done.")

    # 5. Initialize timing
    global last_update_time
    last_update_time = time.time()
    
    # 6. Draw initial frame before showing window
    animation_callback(force=True)
    
    # 7. Enable continuous animation using PyVista's timer
    # This creates a repeating timer that triggers renders, which will call our callback
    def timer_callback(obj, event):
        """Timer callback that updates the animation and renders the scene."""
        if plotter is not None:
            try:
                animation_callback()
                plotter.render()
            except:
                pass
    
    # Observe timer events and create a repeating timer (interval in milliseconds)
    plotter.iren.add_observer("TimerEvent", timer_callback)
    timer_id = plotter.iren.create_timer(int(1000 / FPS), repeating=True)
    
    print("\n--- LAUNCHING INTERACTIVE WINDOW ---")
    print("Close the window to exit.")
    
    # 8. Show the interactive window (blocks until closed)
    plotter.show()

if __name__ == '__main__':
    main()

