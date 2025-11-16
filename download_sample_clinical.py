"""
Download a sample clinical EEG file for testing.
This script attempts to download from various public sources.
"""
import os
import requests
from pathlib import Path
import mne

def download_from_physionet():
    """
    Try to download from PhysioNet using MNE's PhysioNet API.
    """
    try:
        print("Attempting to download from PhysioNet...")
        # CHB-MIT dataset is available on PhysioNet
        # We'll try to get a sample file
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Try using MNE's PhysioNet dataset
        # Note: This requires PhysioNet credentials
        print("Note: PhysioNet requires account registration.")
        print("Visit: https://physionet.org/users/signup/")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def download_sample_from_url():
    """
    Try to download a sample EDF file from a public URL.
    """
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Try a direct download from a public repository
    # Using a sample from a public EEG dataset repository
    urls_to_try = [
        # These are example URLs - we'll need to find actual working ones
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf",
    ]
    
    for url in urls_to_try:
        try:
            print(f"Trying to download from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                filename = data_dir / "sample_clinical.edf"
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded: {filename}")
                return filename
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue
    
    return None

def create_sample_with_mne():
    """
    Create a synthetic but realistic clinical EEG file using MNE.
    This creates a file that mimics clinical data structure.
    """
    print("Creating a synthetic clinical EEG file for testing...")
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create a synthetic raw object with clinical-like properties
    import numpy as np
    
    # Create 23 channels (typical clinical EEG)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2', 'Fpz', 'Oz']
    ch_types = ['eeg'] * len(ch_names)
    
    # Sample at 256 Hz (typical clinical rate)
    sfreq = 256
    duration = 60  # 60 seconds
    n_samples = int(sfreq * duration)
    
    # Generate realistic EEG-like data
    np.random.seed(42)
    data = np.random.randn(len(ch_names), n_samples) * 50e-6  # 50 microvolts
    
    # Add some alpha activity (8-13 Hz) to make it more realistic
    t = np.arange(n_samples) / sfreq
    for i, ch in enumerate(ch_names):
        # Add alpha waves
        alpha_freq = 10 + np.random.randn() * 1
        data[i] += 20e-6 * np.sin(2 * np.pi * alpha_freq * t)
        # Add some temporal lobe activity (for seizure simulation)
        if 'T' in ch:
            data[i] += 30e-6 * np.sin(2 * np.pi * 8 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    # Set standard montage
    try:
        raw.set_montage('standard_1020', on_missing='warn')
    except:
        pass
    
    # Add a seizure annotation at 45 seconds (simulating pre-seizure data)
    onset = [45.0]
    duration = [5.0]
    description = ['seizure']
    annotations = mne.Annotations(onset=onset, duration=duration, description=description)
    raw.set_annotations(annotations)
    
    # Save as EDF using pyedflib
    output_file = data_dir / "sample_clinical.edf"
    
    try:
        import pyedflib
        # Create EDF file
        f = pyedflib.EdfWriter(str(output_file), len(ch_names), file_type=pyedflib.FILETYPE_EDFPLUS)
        
        # Set channel info
        channel_info = []
        data_list = []
        for i, ch_name in enumerate(ch_names):
            ch_dict = {
                'label': ch_name,
                'dimension': 'uV',
                'sample_frequency': sfreq,
                'physical_min': data[i].min() * 1e6,  # Convert to microvolts
                'physical_max': data[i].max() * 1e6,
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            }
            channel_info.append(ch_dict)
            data_list.append(data[i] * 1e6)  # Convert to microvolts
        
        f.setSignalHeaders(channel_info)
        
        # Write data - write all channels at once
        f.writeSamples(data_list)
        
        # Add annotations
        f.writeAnnotation(onset[0], duration[0], description[0])
        
        f.close()
        
        print(f"Created synthetic clinical EEG file: {output_file}")
        print(f"  - {len(ch_names)} channels")
        print(f"  - {duration} seconds duration")
        print(f"  - Seizure annotation at 45 seconds")
        return output_file
    except ImportError:
        print("pyedflib not installed. Installing...")
        os.system("pip install pyedflib")
        # Retry
        import pyedflib
        f = pyedflib.EdfWriter(str(output_file), len(ch_names), file_type=pyedflib.FILETYPE_EDFPLUS)
        channel_info = []
        data_list = []
        for i, ch_name in enumerate(ch_names):
            ch_dict = {
                'label': ch_name,
                'dimension': 'uV',
                'sample_rate': sfreq,
                'physical_min': data[i].min() * 1e6,
                'physical_max': data[i].max() * 1e6,
                'digital_min': -32768,
                'digital_max': 32767,
                'transducer': '',
                'prefilter': ''
            }
            channel_info.append(ch_dict)
            data_list.append(data[i] * 1e6)
        f.setSignalHeaders(channel_info)
        f.writeSamples(data_list)
        f.writeAnnotation(onset[0], duration[0], description[0])
        f.close()
        print(f"Created synthetic clinical EEG file: {output_file}")
        return output_file

if __name__ == '__main__':
    print("=" * 60)
    print("CLINICAL EEG DATA DOWNLOAD")
    print("=" * 60)
    
    # First, try to create a synthetic file (most reliable)
    result = create_sample_with_mne()
    
    if result:
        print(f"\n✅ Success! File created: {result}")
        print("\nNext steps:")
        print("1. Update run_processing.py:")
        print("   USE_CLINICAL_DATA = True")
        print(f"   EDF_FILE_PATH = '{result}'")
        print("   SEIZURE_ONSET_TIME = 45  # or None for auto-detection")
    else:
        print("\n⚠️  Could not create/download clinical data file")
        print("   Please manually download an EDF file and place it in data/")

