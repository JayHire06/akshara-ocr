# data/download_datasets.py
import os
import urllib.request
import zipfile
import tarfile

def download_file(url, output_path):
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

def extract_file(filepath, extract_dir):
    print(f"Extracting {filepath} to {extract_dir}...")
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    print("Extraction complete.")

def organize_dataset():
    """Placeholder to download IIIT-HWS dataset and other open datasets."""
    data_dir = "real_datasets"
    os.makedirs(data_dir, exist_ok=True)
    
    # IIIT-HWS dataset URL (MOCK URL, replace with real URL when available)
    dataset_url = "https://example.com/iiit_hws_dataset.zip" 
    output_file = os.path.join(data_dir, "iiit_hws.zip")
    
    try:
        # download_file(dataset_url, output_file)
        # extract_file(output_file, data_dir)
        print("Implement actual downloading inside `download_datasets.py` using relevant links.")
    except Exception as e:
        print(f"Failed to download/extract: {e}")

if __name__ == '__main__':
    # Example usage
    organize_dataset()
