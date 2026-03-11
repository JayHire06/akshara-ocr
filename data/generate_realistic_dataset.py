import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from download_all_fonts import download_50_hindi_fonts
from synthetic_generator import generate_dataset

def main():
    print("Step 1: Check fonts...")
    fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    download_50_hindi_fonts(fonts_dir=fonts_dir)
    
    output_directory = os.path.join(os.path.dirname(__file__), 'realistic_synthetic')
    
    print("\nStep 2: Starting 50,000 Realistic Dataset Generation...")
    start_time = time.time()
    
    # Generate 50k highly augmented images
    generate_dataset(num_samples=50000, output_dir=output_directory, fonts_dir=fonts_dir, languages=["hi"])
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nGeneration complete in {duration:.2f} seconds.")
    print(f"Total time elapsed: {duration/60:.2f} minutes.")

if __name__ == '__main__':
    main()
