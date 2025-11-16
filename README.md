# Neurobase: A Real-Time Brain Activity Visualizer

Neurobase is a project that translates complex, multi-channel EEG data into an intuitive 3D visualization of the brain's dominant network activity in real time.

## The Vision
Our goal is to move beyond noisy, hard-to-read brainwave data and create a "weather radar for the mind"—a tool that provides immediate, actionable insights for neurologists, researchers, and future brain-computer interfaces.

## The Demo
This animation was generated from real human EEG data. It shows the primary "hub" of brain activity (red sphere) and its functionally connected "spokes" (yellow spheres) shifting from moment to moment.

*(Video gif embed)*

## Tech Stack
- **Language:** Python 3.11
- **Core Libraries:** MNE-Python, NumPy, SciPy
- **Visualization:** PyVista, VTK, PyQt5
- **Animation:** Imageio

## How to Run
1.  Clone the repository.
2.  Set up and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Generate the data: `python run_processing.py`
5.  Generate the animation: `python run_animation.py`
