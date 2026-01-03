import mne
import numpy as np
import pyvista
from functools import lru_cache
from scipy.spatial import cKDTree
from typing import Any, Dict, Tuple

SHOW_BRAIN = True
OUTER_SKIN_SURFACE = 'outer_skin.surf'
DEFAULT_SURFACE_SUBJECT = 'sample'
SENSOR_SURFACE_OFFSET = 0.003  # meters; lifts sensors slightly above mesh
CAMERA_OFFSET = np.array([0.22, -0.25, 0.15])
CAMERA_UP = (0, 0, 1)
HUB_BASE_RADIUS = 0.02
SPOKE_BASE_RADIUS = 0.008


class ActorManager:
    """
    Tracks dynamic PyVista actors (hubs, spokes, connection lines) and animates
    them with fade-in/fade-out transitions instead of popping on/off screen.
    """

    def __init__(self, brain: Any, fade_step: float = 0.25):
        """
        Args:
            brain: Either an ``mne.viz.Brain`` instance or a PyVista plotter. The
                manager extracts the underlying plotter so it can add/remove
                actors directly.
            fade_step: Increment applied to actor opacity every update cycle.
        """
        self.brain = brain
        self.plotter = self._extract_plotter(brain)
        self.fade_step = float(max(fade_step, 0.01))
        self.active_actors: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    @staticmethod
    def _extract_plotter(brain: Any):
        """Returns the PyVista plotter associated with the provided brain."""
        if brain is None:
            return None
        if hasattr(brain, "_renderer") and hasattr(brain._renderer, "plotter"):
            return brain._renderer.plotter
        return brain

    def _add_sphere_actor(self, params: Dict[str, Any]):
        sphere = pyvista.Sphere(
            radius=params["radius"],
            center=params["center"],
            phi_resolution=params.get("phi_resolution", 32),
            theta_resolution=params.get("theta_resolution", 32),
        )
        return self.plotter.add_mesh(
            sphere,
            color=params["color"],
            smooth_shading=params.get("smooth_shading", True),
            specular=params.get("specular", 0.0),
            specular_power=params.get("specular_power", 1.0),
            opacity=0.0,
        )

    def _add_line_actor(self, params: Dict[str, Any]):
        line = pyvista.Line(params["start"], params["end"])
        return self.plotter.add_mesh(
            line,
            color=params["color"],
            line_width=params["line_width"],
            opacity=0.0,
        )

    def _ensure_actor(self, key: Tuple[Any, ...], kind: str, params: Dict[str, Any]):
        """Creates a new actor or refreshes an existing one."""
        actor_state = self.active_actors.get(key)
        target_opacity = float(np.clip(params.get("target_opacity", 1.0), 0.0, 1.0))

        if actor_state is None:
            try:
                actor = (
                    self._add_sphere_actor(params)
                    if kind in ("hub", "spoke")
                    else self._add_line_actor(params)
                )
            except Exception:
                return

            self.active_actors[key] = {
                "actor_object": actor,
                "opacity": 0.0,
                "state": "fading_in",
                "kind": kind,
                "target_opacity": target_opacity,
                "line_width": params.get("line_width"),
            }
            return

        actor_state["target_opacity"] = target_opacity
        actor_state["line_width"] = params.get("line_width", actor_state.get("line_width"))

        actor = actor_state["actor_object"]
        if kind == "line" and actor_state.get("line_width"):
            try:
                actor.GetProperty().SetLineWidth(actor_state["line_width"])
            except Exception:
                pass

    def _remove_actor(self, actor):
        remover = None
        if self.plotter is not None and hasattr(self.plotter, "remove_actor"):
            remover = self.plotter.remove_actor
        elif hasattr(self.brain, "remove_actor"):
            remover = self.brain.remove_actor
        if remover is None or actor is None:
            return
        try:
            remover(actor)
        except Exception:
            pass

    def _update_opacities(self):
        keys_to_delete = []
        for key, actor_state in self.active_actors.items():
            actor = actor_state.get("actor_object")
            if actor is None:
                keys_to_delete.append(key)
                continue

            state = actor_state.get("state", "visible")
            target = actor_state.get("target_opacity", 1.0)

            if state == "fading_in":
                actor_state["opacity"] = min(
                    target, actor_state["opacity"] + self.fade_step
                )
                if abs(actor_state["opacity"] - target) <= 1e-3:
                    actor_state["state"] = "visible"
            elif state == "fading_out":
                actor_state["opacity"] = max(0.0, actor_state["opacity"] - self.fade_step)
            else:  # visible
                diff = target - actor_state["opacity"]
                if abs(diff) > 1e-3:
                    step = np.sign(diff) * min(abs(diff), self.fade_step)
                    actor_state["opacity"] = float(
                        np.clip(actor_state["opacity"] + step, 0.0, 1.0)
                    )

            try:
                actor.GetProperty().SetOpacity(actor_state["opacity"])
            except Exception:
                pass

            if actor_state["state"] == "fading_out" and actor_state["opacity"] <= 0.0:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            state = self.active_actors.pop(key, None)
            if state and state.get("actor_object") is not None:
                self._remove_actor(state["actor_object"])

    def update(
        self,
        hub_idx: int,
        spoke_indices,
        sensor_locs: np.ndarray,
        spoke_strengths: Dict[int, float] | None = None,
        hub_strength: float = 1.0,
    ):
        """
        Synchronizes on-screen actors with the current frame's desired hub/spoke set.

        Args:
            hub_idx: Index of the hub electrode for the current frame.
            spoke_indices: Iterable of spoke indices to keep visible.
            sensor_locs: Array of electrode coordinates.
            spoke_strengths: Optional dict mapping spoke index -> strength (0-1).
            hub_strength: Strength (0-1) controlling hub size/opacity.
        """
        if self.plotter is None or sensor_locs is None or len(sensor_locs) == 0:
            return

        spoke_strengths = spoke_strengths or {}
        if spoke_indices is None:
            spoke_indices_list = []
        else:
            spoke_indices_list = list(spoke_indices)

        desired: Dict[Tuple[Any, ...], Tuple[str, Dict[str, Any]]] = {}

        hub_strength = float(np.clip(hub_strength, 0.0, 1.0))
        hub_center = sensor_locs[hub_idx]
        hub_radius = HUB_BASE_RADIUS
        hub_opacity = 0.6 + 0.35 * hub_strength
        desired[("hub", hub_idx)] = (
            "hub",
            {
                "center": hub_center,
                "radius": hub_radius,
                "color": "red",
                "specular": 0.6,
                "specular_power": 35,
                "smooth_shading": True,
                "phi_resolution": 32,
                "theta_resolution": 32,
                "target_opacity": float(np.clip(hub_opacity, 0.0, 1.0)),
            },
        )

        for spoke_idx in spoke_indices_list:
            strength = float(np.clip(spoke_strengths.get(spoke_idx, 0.0), 0.0, 1.0))
            spoke_center = sensor_locs[spoke_idx]
            radius = SPOKE_BASE_RADIUS
            opacity = 0.35 + 0.45 * strength
            desired[("spoke", spoke_idx)] = (
                "spoke",
                {
                    "center": spoke_center,
                    "radius": radius,
                    "color": "yellow",
                    "specular": 0.55,
                    "specular_power": 35,
                    "smooth_shading": True,
                    "phi_resolution": 28,
                    "theta_resolution": 28,
                    "target_opacity": float(np.clip(opacity, 0.0, 1.0)),
                },
            )

            line_width = 2 + 4 * strength
            line_opacity = 0.35 + 0.55 * strength
            desired[("line", hub_idx, spoke_idx)] = (
                "line",
                {
                    "start": hub_center,
                    "end": spoke_center,
                    "color": "cyan",
                    "line_width": line_width,
                    "target_opacity": float(np.clip(line_opacity, 0.0, 1.0)),
                },
            )

        desired_keys = set(desired.keys())
        for key, state in self.active_actors.items():
            if key not in desired_keys and state.get("state") != "fading_out":
                state["state"] = "fading_out"
                state["target_opacity"] = 0.0

        for key, (kind, params) in desired.items():
            self._ensure_actor(key, kind, params)
            state = self.active_actors.get(key)
            if state and state["state"] == "fading_out":
                state["state"] = "visible"

        self._update_opacities()

# Temporal lobe channel names in standard 10-20 system
TEMPORAL_LOBE_CHANNELS = [
    'T7', 'T8', 'T9', 'T10',  # Temporal
    'TP9', 'TP10',  # Temporal-Parietal
    'FT9', 'FT10',  # Frontal-Temporal
    'TP7', 'TP8',  # Temporal-Parietal (alternative naming)
    'FT7', 'FT8',  # Frontal-Temporal (alternative naming)
    'T3', 'T4', 'T5', 'T6',  # Old naming convention
]

@lru_cache(maxsize=1)
def _load_temporal_lobe_mesh():
    """
    Loads the temporal lobe mesh from Freesurfer parcellations.
    
    Returns:
        tuple[pyvista.PolyData | None, np.ndarray | None]:
            (temporal_lobe_mesh, vertex_coordinates)
    """
    try:
        subjects_dir = mne.datasets.sample.data_path() / 'subjects'
        subject = DEFAULT_SURFACE_SUBJECT
        
        # Load parcellation labels - try different parcellation schemes
        parc_schemes = ['aparc.a2009s', 'aparc', 'aparc.DKTatlas']
        labels = None
        
        for parc in parc_schemes:
            try:
                labels = mne.read_labels_from_annot(
                    subject, 
                    parc=parc, 
                    subjects_dir=subjects_dir,
                    verbose=False
                )
                if labels:
                    break
            except Exception:
                continue
        
        if not labels:
            return None, None
        
        # Find temporal lobe labels (both hemispheres)
        temporal_labels = []
        temporal_keywords = ['temporal', 'superiortemporal', 'middletemporal', 'inferiortemporal']
        
        for label in labels:
            label_name_lower = label.name.lower()
            if any(keyword in label_name_lower for keyword in temporal_keywords):
                temporal_labels.append(label)
        
        if not temporal_labels:
            return None, None
        
        # Load brain surfaces for both hemispheres
        all_temporal_faces = []
        all_temporal_coords = []
        coord_offset = 0
        
        for hemi in ['lh', 'rh']:
            # Try pial surface first, then white matter
            surf_path = subjects_dir / subject / 'surf' / f'{hemi}.pial'
            if not surf_path.exists():
                surf_path = subjects_dir / subject / 'surf' / f'{hemi}.white'
            if not surf_path.exists():
                continue
            
            coords, faces = mne.read_surface(surf_path, verbose=False)
            coords = np.asarray(coords, dtype=float)
            
            # Scale coordinates if needed
            max_abs = np.max(np.abs(coords))
            if max_abs > 5.0:
                coords = coords / 1000.0
            
            # Get temporal lobe vertices for this hemisphere
            hemi_temporal_vertices = set()
            for label in temporal_labels:
                if label.hemi == hemi:
                    hemi_temporal_vertices.update(label.vertices)
            
            if not hemi_temporal_vertices:
                continue
            
            hemi_temporal_vertices = np.array(sorted(hemi_temporal_vertices))
            
            # Create mask for temporal lobe vertices
            vertex_mask = np.zeros(len(coords), dtype=bool)
            vertex_mask[hemi_temporal_vertices] = True
            
            # Extract faces that belong to temporal lobe
            hemi_temporal_faces = []
            for face in faces:
                if any(vertex_mask[v] for v in face):
                    hemi_temporal_faces.append(face)
            
            if not hemi_temporal_faces:
                continue
            
            # Get unique vertices used in temporal faces
            unique_vertices = np.unique(np.array(hemi_temporal_faces).ravel())
            vertex_map = {old_idx: new_idx + coord_offset for new_idx, old_idx in enumerate(unique_vertices)}
            
            # Remap faces and add to combined list
            remapped_faces = np.array([[vertex_map[v] for v in face] for face in hemi_temporal_faces])
            all_temporal_faces.append(remapped_faces)
            all_temporal_coords.append(coords[unique_vertices])
            
            coord_offset += len(unique_vertices)
        
        if not all_temporal_faces:
            return None, None
        
        # Combine coordinates and faces from both hemispheres
        temporal_coords = np.vstack(all_temporal_coords)
        temporal_faces = np.vstack(all_temporal_faces)
        
        # Create PyVista mesh
        faces_pv = np.column_stack([np.full(len(temporal_faces), 3), temporal_faces]).astype(np.int32)
        mesh = pyvista.PolyData(temporal_coords, faces_pv.ravel())
        mesh = mesh.compute_normals(inplace=False)
        
        return mesh, temporal_coords
        
    except Exception as e:
        print(f"Warning: Could not load temporal lobe mesh: {e}")
        return None, None

@lru_cache(maxsize=1)
def _load_brain_surface():
    """
    Loads and caches the outer skin surface as a PyVista mesh.

    Returns:
        tuple[pyvista.PolyData | None, np.ndarray | None, np.ndarray | None]:
            (brain_mesh, vertex_coordinates, vertex_normals)
    """
    try:
        subjects_dir = mne.datasets.sample.data_path() / 'subjects'
        surf_path = subjects_dir / DEFAULT_SURFACE_SUBJECT / 'bem' / OUTER_SKIN_SURFACE
        if not surf_path.exists():
            return None, None, None

        coords, faces = mne.read_surface(surf_path, verbose=False)
        coords = np.asarray(coords, dtype=float)
        max_abs = np.max(np.abs(coords))
        if max_abs > 5.0:
            coords = coords / 1000.0
        faces_pv = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int32)
        mesh = pyvista.PolyData(coords, faces_pv.ravel())
        mesh = mesh.compute_normals(inplace=False)
        normals = mesh.point_normals
        return mesh, coords, normals
    except Exception as e:
        # Print error for debugging but don't crash
        print(f"Warning: Could not load brain surface: {e}")
        return None, None, None

def get_lobe_sensors(info, lobe_name="temporal", sensor_locs=None):
    """
    Identifies which EEG sensors are located over a specific brain lobe.
    
    Uses channel names first (for standard 10-20 naming), then falls back to
    spatial position if channel names don't match (for datasets with generic names).
    
    Args:
        info (mne.Info): The MNE info object containing channel information.
        lobe_name (str): Name of the lobe to identify. Currently supports "temporal".
        sensor_locs (np.ndarray | None): Optional array of sensor 3D positions (n_sensors, 3).
            If provided and channel name matching fails, spatial position will be used.
    
    Returns:
        list[int]: List of integer indices for channels in the specified lobe.
    """
    if lobe_name.lower() != "temporal":
        raise ValueError(f"Currently only 'temporal' lobe is supported, got '{lobe_name}'")
    
    channel_names = info.ch_names
    lobe_indices = []
    
    # First, try to match by channel names (standard 10-20 system)
    for idx, ch_name in enumerate(channel_names):
        # Normalize channel name (remove spaces, convert to uppercase)
        ch_upper = ch_name.upper().strip()
        
        # Check if this channel matches any temporal lobe pattern
        for temporal_pattern in TEMPORAL_LOBE_CHANNELS:
            if ch_upper == temporal_pattern.upper():
                lobe_indices.append(idx)
                break
            # Also check for partial matches (e.g., "T7" in "EEG T7-REF")
            if temporal_pattern.upper() in ch_upper:
                lobe_indices.append(idx)
                break
    
    # If no matches found by name and sensor locations provided, use spatial position
    if len(lobe_indices) == 0 and sensor_locs is not None and len(sensor_locs) > 0:
        # Temporal lobe is typically on the sides of the head (high absolute X values)
        # and in the middle-to-lower part of the head (moderate Y, moderate Z)
        sensor_locs = np.asarray(sensor_locs)
        
        # Calculate statistics for spatial filtering
        x_coords = sensor_locs[:, 0]
        y_coords = sensor_locs[:, 1]
        z_coords = sensor_locs[:, 2]
        
        # Temporal lobe sensors are typically:
        # - On the sides: |X| > threshold (e.g., 60th percentile of |X|)
        # - Not too far forward/back: Y within reasonable range
        # - Not too high/low: Z within reasonable range
        
        abs_x = np.abs(x_coords)
        x_threshold = np.percentile(abs_x, 60)  # Top 40% by absolute X
        
        # Find sensors with high absolute X (sides of head)
        for idx in range(len(sensor_locs)):
            if abs_x[idx] >= x_threshold:
                lobe_indices.append(idx)
    
    return lobe_indices

def project_sensors_to_surface(sensor_locs: np.ndarray, offset: float = SENSOR_SURFACE_OFFSET) -> np.ndarray:
    """
    Repositions sensors so they sit on the outer scalp surface without collapsing
    onto a single vertex.

    The approach keeps each electrode's angular position but rescales it so every
    point lies slightly outside the brain mesh radius.

    Args:
        sensor_locs: Array of sensor positions (n_sensors, 3).
        offset: Extra distance (meters) to push sensors beyond the scalp.

    Returns:
        np.ndarray: Adjusted sensor coordinates aligned to the scalp mesh.
    """
    brain_mesh, coords, normals = _load_brain_surface()

    sensor_center = np.mean(sensor_locs, axis=0)
    base_center = sensor_center
    adjusted = sensor_locs.copy()

    if coords is not None and len(coords) > 0:
        mesh_center = np.mean(coords, axis=0)
        base_center = mesh_center
        adjusted = sensor_locs - sensor_center + mesh_center
        try:
            tree = cKDTree(coords)
            _, nearest_idx = tree.query(adjusted, k=1)
            projected = coords[nearest_idx]
            if normals is not None and len(normals) == len(coords):
                normal_vectors = normals[nearest_idx]
                normal_norms = np.linalg.norm(normal_vectors, axis=1, keepdims=True)
                safe_normals = np.where(normal_norms == 0, 1.0, normal_norms)
                unit_normals = normal_vectors / safe_normals
            else:
                fallback_vectors = projected - base_center
                fallback_norms = np.linalg.norm(fallback_vectors, axis=1, keepdims=True)
                safe_fallback = np.where(fallback_norms == 0, 1.0, fallback_norms)
                unit_normals = fallback_vectors / safe_fallback
            elevated = projected + unit_normals * offset
            return elevated
        except Exception:
            pass

    vectors = adjusted - base_center
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    unit_dirs = vectors / safe_norms
    elevated = base_center + unit_dirs * (safe_norms + offset)
    return elevated

def plot_brain_frame(
    info,
    hub_idx,
    spoke_indices,
    sensor_locs,
    screenshot_path,
    frame_metadata=None,
    spoke_strengths=None,
    hub_strength=1.0,
    temporal_sensor_indices=None,
):
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
        frame_metadata (dict | None): Optional metadata for on-screen overlays.
        spoke_strengths (dict | None): Per-spoke visibility weights (0-1) used for
            fading spokes in/out.
        hub_strength (float): Visibility weight (0-1) for the current hub sphere.
        temporal_sensor_indices (list | None): List of sensor indices in the temporal lobe.
            If provided and hub is in this list, the temporal lobe will be highlighted.
    
    Returns:
        None
    """
    plotter = None
    
    try:
        # Create a PyVista plotter directly (no MNE involvement)
        # Window size must be divisible by 16 for video encoding (1920x1088 = 120*16 x 68*16)
        plotter = pyvista.Plotter(off_screen=True, window_size=[1920, 1088])
        plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

        # Check if hub is in temporal lobe
        is_hub_in_temporal = False
        if temporal_sensor_indices is not None:
            is_hub_in_temporal = hub_idx in temporal_sensor_indices

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
        
        # Plot spokes first as smaller, yellow spheres
        if spoke_indices:
            spoke_strengths = spoke_strengths or {}
            spoke_locs = sensor_locs[spoke_indices]
            for spoke_idx, spoke_loc in zip(spoke_indices, spoke_locs):
                strength = float(np.clip(spoke_strengths.get(spoke_idx, 0.0), 0.0, 1.0))
                radius = 0.006 + 0.004 * strength
                opacity = 0.35 + 0.45 * strength
                sphere = pyvista.Sphere(radius=radius, center=spoke_loc, phi_resolution=28, theta_resolution=28)
                plotter.add_mesh(
                    sphere,
                    color='yellow',
                    smooth_shading=True,
                    opacity=opacity,
                    specular=0.55,
                    specular_power=35,
                )

        # Plot the hub as a MUCH larger, red sphere so it's clearly visible
        if hub_idx is not None:
            hub_loc = sensor_locs[hub_idx]
            # Hub needs to be significantly larger than spokes
            hub_strength = float(np.clip(hub_strength, 0.0, 1.0))
            hub_radius = 0.012 + 0.010 * hub_strength
            hub_opacity = 0.6 + 0.35 * hub_strength
            hub_sphere = pyvista.Sphere(radius=hub_radius, center=hub_loc, phi_resolution=32, theta_resolution=32)
            plotter.add_mesh(
                hub_sphere,
                color='red',
                smooth_shading=True,
                opacity=hub_opacity,
                specular=0.6,
                specular_power=35,
            )
            
            # Add the connection lines with increased width for visibility
            for spoke_idx in spoke_indices:
                spoke_loc = sensor_locs[spoke_idx]
                line = pyvista.Line(hub_loc, spoke_loc)
                raw_strength = spoke_strengths.get(spoke_idx, 0.0) if spoke_strengths else 1.0
                strength = float(np.clip(raw_strength, 0.0, 1.0))
                line_width = 2 + 4 * strength
                opacity = 0.35 + 0.55 * strength
                plotter.add_mesh(line, color='cyan', line_width=line_width, opacity=opacity)
        
        # Add temporal lobe highlight if hub is in temporal region
        if is_hub_in_temporal:
            # Try to load the actual temporal lobe mesh from Freesurfer
            temporal_mesh, temporal_coords = _load_temporal_lobe_mesh()
            
            if temporal_mesh is not None and temporal_coords is not None:
                # Use the actual temporal lobe mesh
                plotter.add_mesh(
                    temporal_mesh,
                    color='red',
                    opacity=0.35,
                    smooth_shading=True,
                    ambient=0.5,
                    specular=0.2,
                    name="temporal_lobe_mesh"
                )
            else:
                # Fallback to sphere highlight if mesh loading fails
                hub_loc = sensor_locs[hub_idx]
                highlight_radius = 0.08  # Radius in meters for the highlight sphere
                highlight_sphere = pyvista.Sphere(
                    radius=highlight_radius,
                    center=hub_loc,
                    theta_resolution=60,
                    phi_resolution=60,
                )
                plotter.add_mesh(
                    highlight_sphere,
                    color='red',
                    opacity=0.25,
                    smooth_shading=True,
                    ambient=0.6,
                    specular=0.1,
                    name="temporal_highlight"
                )
        
        # NOW add the brain mesh LAST so sensors render in front/through it
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
        
        # Set the camera view for a nice angle (fixed per-frame)
        focus_point = brain_center if np.any(brain_center) else np.array([0.0, 0.0, 0.0])
        camera_pos = focus_point + CAMERA_OFFSET
        plotter.camera_position = [
            tuple(camera_pos.tolist()),
            tuple(focus_point.tolist()),
            CAMERA_UP,
        ]
        plotter.camera.zoom(0.85)
        plotter.reset_camera_clipping_range()

        if frame_metadata:
            frame_idx = frame_metadata.get('frame_index')
            timestamp = frame_metadata.get('timestamp_seconds')
            hub_label = frame_metadata.get('hub_label', hub_idx)
            spoke_count = frame_metadata.get('spoke_count', len(spoke_indices))
            fps = frame_metadata.get('fps')
            hub_coord_mm = frame_metadata.get('hub_coord_mm')
            mean_spoke_strength = frame_metadata.get('mean_spoke_strength')
            max_spoke_strength = frame_metadata.get('max_spoke_strength')

            overlay_lines = [
                f"Frame: {frame_idx:04d}" if frame_idx is not None else None,
                f"Time: {timestamp:05.2f}s" if timestamp is not None else None,
                f"Hub: {hub_label}",
                (
                    f"Hub xyz (mm): {hub_coord_mm[0]:6.1f}, {hub_coord_mm[1]:6.1f}, {hub_coord_mm[2]:6.1f}"
                    if hub_coord_mm is not None
                    else None
                ),
                f"Spokes: {spoke_count}",
                (
                    f"Mean strength: {mean_spoke_strength:0.2f} / max {max_spoke_strength:0.2f}"
                    if mean_spoke_strength is not None and max_spoke_strength is not None
                    else None
                ),
                f"FPS: {fps}" if fps is not None else None,
            ]
            overlay_text = "\n".join([line for line in overlay_lines if line is not None])
            plotter.add_text(overlay_text, position='upper_left', font_size=20, color='white', shadow=True)
        
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
