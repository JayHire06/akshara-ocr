import os
import urllib.request
import time
import sys

# Ensure data package can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from data.synthetic_generator import generate_dataset

def download_hindi_fonts():
    fonts_to_download = {
        "NotoSansDevanagari-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
        "TiroDevanagariHindi-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf",
        "YatraOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/yatraone/YatraOne-Regular.ttf",
        "RozhaOne-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/rozhaone/RozhaOne-Regular.ttf",
        "Amita-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/amita/Amita-Regular.ttf",
        "Arya-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/arya/Arya-Regular.ttf",
        "Karma-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/karma/Karma-Regular.ttf",
        "Eczar-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/eczar/Eczar%5Bwght%5D.ttf",
        "Glegoo-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/glegoo/Glegoo-Regular.ttf",
        "Halant-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/halant/Halant-Regular.ttf"
    }

    fonts_dir = os.path.join(os.path.dirname(__file__), 'data', 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    
    print(f"Downloading {len(fonts_to_download)} Hindi fonts...")
    for name, url in fonts_to_download.items():
        font_path = os.path.join(fonts_dir, name)
        if not os.path.exists(font_path):
            try:
                urllib.request.urlretrieve(url, font_path)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            print(f"Font {name} already exists.")

def get_dir_size(path="."):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

if __name__ == '__main__':
    download_hindi_fonts()
    
    # Run Generation
    print("\nStarting dataset generation (1000 samples)...")
    start_time = time.time()
    
    output_directory = os.path.join(os.path.dirname(__file__), 'data', 'train')
    os.makedirs(output_directory, exist_ok=True)
    
    fonts_dir = os.path.join(os.path.dirname(__file__), 'data', 'fonts')
    
    # Generate 1000 Hindi samples
    generate_dataset(num_samples=1000, output_dir=output_directory, fonts_dir=fonts_dir, languages=["hi"])
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Generation took: {duration:.2f} seconds")
    
    # Verify outputs
    size_bytes = get_dir_size(output_directory)
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nSize of {output_directory} on disk: {size_mb:.2f} MB")
    
    # Print sample file paths and labels
    labels_csv = os.path.join(output_directory, "labels.csv")
    print(f"\nSample contents of {labels_csv}:")
    with open(labels_csv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())
            if i >= 5:  # header + 5 rows
                break
