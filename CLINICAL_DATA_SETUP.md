# Clinical Data Setup Guide

## Overview
This guide explains how to use the neural-hub-finder with real clinical epilepsy EEG data.

## Option 1: Using OpenNeuro Datasets

### Downloading from OpenNeuro

1. **Install datalad** (if not already installed):
   ```bash
   pip install datalad
   ```

2. **Download a dataset**:
   ```bash
   python download_clinical_data.py
   ```
   
   Or manually using datalad:
   ```bash
   datalad install https://github.com/OpenNeuroDatasets/ds003029.git data/ds003029
   cd data/ds003029
   datalad get sub-01/ses-presurgery/eeg/*.edf
   ```

3. **Find seizure annotations**:
   - Check the dataset's `events.tsv` or annotation files
   - Look for seizure onset markers in the EDF file annotations
   - Common annotation channels: `STIM`, `TRIGGER`, or event markers

4. **Update `run_processing.py`**:
   ```python
   USE_CLINICAL_DATA = True
   EDF_FILE_PATH = 'data/ds003029/sub-01/ses-presurgery/eeg/sub-01_ses-presurgery_task-ictal_eeg.edf'
   SEIZURE_ONSET_TIME = 300  # Update based on your data's annotations
   CLIP_DURATION = 30
   ```

## Option 2: Using Your Own EDF File

1. **Place your EDF file** in the `data/` directory:
   ```bash
   cp your_file.edf data/
   ```

2. **Identify seizure onset time**:
   - Open the EDF file in a viewer (e.g., EDFBrowser, MNE-Python)
   - Find the seizure onset marker
   - Note the time in seconds

3. **Update `run_processing.py`**:
   ```python
   USE_CLINICAL_DATA = True
   EDF_FILE_PATH = 'data/your_file.edf'
   SEIZURE_ONSET_TIME = <your_seizure_onset_time>
   CLIP_DURATION = 30
   ```

4. **Run processing**:
   ```bash
   python run_processing.py
   python run_animation.py
   ```

## Recommended OpenNeuro Datasets

- **ds003029**: SEEG Epilepsy dataset (large, requires significant download)
- **ds003505**: CHB-MIT Scalp EEG Database (smaller, well-documented)
- **ds004148**: TUH EEG Corpus (comprehensive, includes seizure annotations)

## Notes

- The code automatically detects temporal lobe sensors using:
  1. Standard 10-20 channel names (T7, T8, TP9, etc.)
  2. Spatial position fallback (for generic channel names)
  
- Temporal lobe highlighting will automatically activate when hubs are detected in temporal regions

- The visualization uses Freesurfer parcellations to show the actual temporal lobe anatomy

