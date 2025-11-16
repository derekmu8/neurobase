"""
Script to download clinical epilepsy EEG data from OpenNeuro.
"""
import os
from pathlib import Path

def download_openneuro_dataset(dataset_id, output_dir='data'):
    """
    Download a dataset from OpenNeuro using datalad.
    
    Args:
        dataset_id: OpenNeuro dataset ID (e.g., 'ds003029')
        output_dir: Directory to save the data
    """
    try:
        import datalad.api as dl
    except ImportError:
        print("datalad not installed. Installing...")
        os.system("pip install datalad")
        import datalad.api as dl
    
    output_path = Path(output_dir) / dataset_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_url = f"https://github.com/OpenNeuroDatasets/{dataset_id}.git"
    
    print(f"Downloading dataset {dataset_id} from OpenNeuro...")
    print(f"This may take a while depending on dataset size...")
    
    try:
        # Clone the dataset
        dataset = dl.install(source=dataset_url, path=str(output_path))
        print(f"Dataset installed to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: You can manually download from:")
        print(f"https://openneuro.org/datasets/{dataset_id}")
        return None

if __name__ == '__main__':
    # Try to download the SEEG Epilepsy dataset
    # Note: This is a large dataset, so we'll download just one subject
    dataset_id = 'ds003029'
    data_dir = download_openneuro_dataset(dataset_id)
    
    if data_dir:
        print(f"\nData downloaded to: {data_dir}")
        print("\nNext steps:")
        print("1. Find an EDF file with seizure annotations")
        print("2. Update run_processing.py with the file path and seizure onset time")
        print("3. Set USE_CLINICAL_DATA = True in run_processing.py")

