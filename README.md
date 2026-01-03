# Neurobase

**Real-time brain network visualization from EEG data.**

Neurobase transforms multi-channel EEG signals into an intuitive 3D visualization showing the brain's dominant "hub" of activity and its connected regions — like a weather radar for neural activity.

![Neural Hub Animation](neural_hubs.mp4)

## What It Does

1. **Loads clinical EEG data** from PhysioNet (CHB-MIT seizure dataset)
2. **Filters alpha band** (8-13 Hz) activity
3. **Computes power envelopes** using Hilbert transform
4. **Identifies hub/spoke networks** — the dominant activity center and connected regions
5. **Renders 3D brain visualization** with smooth transitions

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/neurobase.git
cd neurobase
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python run_processing.py   # Process EEG → hub_data.npy
python run_animation.py    # Generate video → neural_hubs.mp4
```

## Tech Stack

| Component | Libraries |
|-----------|-----------|
| Signal Processing | MNE-Python, SciPy, NumPy |
| Visualization | PyVista, VTK |
| Video | Imageio |

## Project Structure

```
neurobase/
├── data_pipeline.py      # EEG loading & preprocessing
├── feature_pipeline.py   # Power envelope calculation
├── logic.py              # Hub/spoke detection algorithm
├── post_processing.py    # Temporal smoothing
├── visualizer.py         # 3D brain rendering
├── run_processing.py     # Main data pipeline
└── run_animation.py      # Video generation
```

## License

MIT
