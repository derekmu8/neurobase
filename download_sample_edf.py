"""
Download a sample epilepsy EDF file directly from OpenNeuro.
This is a simpler approach that doesn't require git-annex.
"""
import os
import requests
from pathlib import Path
import zipfile

def download_chbmit_sample():
    """
    Download a sample file from CHB-MIT dataset.
    Note: This downloads a single file for testing purposes.
    """
    # CHB-MIT dataset files are available via direct download
    # Using a sample file from the dataset
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # For now, we'll create a note about manual download
    # since direct S3 access requires authentication
    print("=" * 60)
    print("CLINICAL DATA DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nOption 1: Manual Download from OpenNeuro")
    print("-" * 60)
    print("1. Visit: https://openneuro.org/datasets/ds003505")
    print("2. Click 'Download' and select a subject (e.g., chb01)")
    print("3. Download a single EDF file (e.g., chb01_01.edf)")
    print("4. Place it in the 'data/' directory")
    print("\nOption 2: Use Datalad (requires git-annex)")
    print("-" * 60)
    print("Install git-annex first:")
    print("  macOS: brew install git-annex")
    print("  Then: datalad install https://github.com/OpenNeuroDatasets/ds003505.git data/chbmit")
    print("\nOption 3: Use a smaller test dataset")
    print("-" * 60)
    print("For testing, you can use any EDF file with EEG data.")
    print("Place it in data/ and update run_processing.py")
    print("=" * 60)
    
    # Check if any EDF files already exist
    edf_files = list(data_dir.glob('*.edf'))
    if edf_files:
        print(f"\n✅ Found existing EDF file(s):")
        for edf in edf_files:
            print(f"   - {edf}")
        return edf_files[0]
    else:
        print("\n⚠️  No EDF files found in data/ directory")
        print("   Please download a clinical EDF file and place it in data/")
        return None

if __name__ == '__main__':
    download_chbmit_sample()

