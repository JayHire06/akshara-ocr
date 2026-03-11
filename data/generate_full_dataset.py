import os
import sys
import time
from download_all_fonts import download_50_hindi_fonts

# Ensure synthetic_generator can be imported
sys.path.append(os.path.dirname(__file__))
from synthetic_generator import generate_dataset

def main():
    print("Step 1: Ensuring 50+ Devanagari fonts are available...")
    download_50_hindi_fonts(fonts_dir="fonts")
    
    print("\nStep 2: Starting 500,000 Dataset Generation...")
    start_time = time.time()
    
    output_directory = os.path.join(os.path.dirname(__file__))
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    
    # Generate 500,000 images, splitted 80/10/10
    generate_dataset(num_samples=500000, output_dir=output_directory, fonts_dir=fonts_dir, languages=["hi"])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nGeneration complete in {duration:.2f} seconds.")
    print(f"Total time elapsed: {duration/60:.2f} minutes.")
    
    # We can also compute sizes
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_directory, split)
        if os.path.exists(split_dir):
            size_bytes = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(split_dir) for f in filenames)
            print(f"Size of {split}: {size_bytes / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    main()
