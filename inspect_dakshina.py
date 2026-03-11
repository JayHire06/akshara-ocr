import os
import urllib.request
import tarfile

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "real_datasets")
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, "dakshina_dataset_v1.0.tar")
    extract_path = os.path.join(data_dir, "dakshina")
    
    if not os.path.exists(tar_path):
        print("Downloading Dakshina dataset...")
        urllib.request.urlretrieve("https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar", tar_path)
        print("Downloaded.")
    
    if not os.path.exists(os.path.join(extract_path, "dakshina_dataset_v1.0")):
        print("Extracting dataset...")
        os.makedirs(extract_path, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)
            
    print("Extraction complete. Analyzing Hindi lexicon...")
    hi_lexicon_dir = os.path.join(extract_path, "dakshina_dataset_v1.0", "hi", "lexicon")
    
    if not os.path.exists(hi_lexicon_dir):
        print(f"Path not found: {hi_lexicon_dir}")
        return
        
    files = os.listdir(hi_lexicon_dir)
    print("Files in hi/lexicon/:", files)
    
    # We will read a bit of the files to see what format it is
    for f in files:
        if f.endswith('.txt') or f.endswith('.tsv') or f.endswith('.csv'):
            print(f"--- Sample of {f} ---")
            with open(os.path.join(hi_lexicon_dir, f), 'r', encoding='utf-8') as file:
                for i in range(5):
                    print(file.readline().strip())
    
if __name__ == "__main__":
    main()
