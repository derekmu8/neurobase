import mne
import numpy as np
import pyvista
from pathlib import Path

def plot_brain_frame(info, hub_idx, spoke_indices, sensor_locs, screenshot_path):
    """
    Plots a single frame of the brain animation and saves it as an image.
    
    This function bypasses MNE's plot_alignment to avoid macOS segmentation faults.
    Instead, it directly loads brain surfaces using PyVista and renders everything
    with full control over the VTK pipeline.
    
    Uses off-screen rendering to avoid popup windows.

    Args:
        info (mne.Info): The MNE info object containing sensor information.
        hub_idx (int): The index of the hub sensor.
        spoke_indices (list): A list of indices for the spoke sensors.
        sensor_locs (np.ndarray): Array of 3D coordinates for all sensors.
        screenshot_path (str): Path where the screenshot should be saved.
    
    Returns:
        None
    """
    plotter = None
    
    try:
        # Create a PyVista plotter directly (no MNE involvement)
        # Window size must be divisible by 16 for video encoding (1920x1088 = 120*16 x 68*16)
        plotter = pyvista.Plotter(off_screen=True, window_size=[1920, 1088])
        
        # Load brain surface meshes directly from FreeSurfer files
        subjects_dir = mne.datasets.sample.data_path() / 'subjects'
        subject = 'sample'
        
        # Try to load the outer skin (scalp) surface
        try:
            surf_path = subjects_dir / subject / 'bem' / 'outer_skin.surf'
            if surf_path.exists():
                # Read FreeSurfer surface using MNE, then convert to PyVista mesh
                coords, faces = mne.read_surface(surf_path, verbose=False)
                # Convert faces from 0-indexed to the format PyVista expects
                faces_pv = np.column_stack([np.full(len(faces), 3), faces])
                brain_mesh = pyvista.PolyData(coords, faces_pv.ravel())
                plotter.add_mesh(brain_mesh, color='lightgray', opacity=0.3, smooth_shading=True)
        except Exception as e:
            # If we can't load the brain surface, just continue without it
            # The sensors and connections will still be visible
            pass
        
        # Plot spokes first as smaller, yellow spheres
        if spoke_indices:
            spoke_locs = sensor_locs[spoke_indices]
            # Create sphere actors for each spoke
            for spoke_loc in spoke_locs:
                sphere = pyvista.Sphere(radius=0.003, center=spoke_loc)
                plotter.add_mesh(sphere, color='yellow', smooth_shading=True)

        # Plot the hub as a larger, red sphere so it's clearly visible
        hub_loc = sensor_locs[hub_idx]
        hub_sphere = pyvista.Sphere(radius=0.006, center=hub_loc)
        plotter.add_mesh(hub_sphere, color='red', smooth_shading=True)
        
        # Add the connection lines
        for spoke_idx in spoke_indices:
            spoke_loc = sensor_locs[spoke_idx]
            line = pyvista.Line(hub_loc, spoke_loc)
            plotter.add_mesh(line, color='cyan', line_width=3)
        
        # Set background color
        plotter.set_background('black')
        
        # Set the camera view for a nice angle
        plotter.camera_position = 'xy'  # Start with a good default view
        plotter.camera.azimuth = 45
        plotter.camera.elevation = 20
        plotter.camera.zoom(1.3)
        
        # Save the screenshot
        plotter.screenshot(screenshot_path)
        
    finally:
        # Ensure proper cleanup even if an error occurs
        # This prevents memory leaks by explicitly closing and destroying the plotter
        if plotter is not None:
            try:
                plotter.close()
                del plotter  # Explicit deletion to help garbage collection
            except Exception:
                pass  # Ignore errors during cleanup
