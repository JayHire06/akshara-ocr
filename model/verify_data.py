import os
from PIL import Image
import numpy as np

TRAIN_LABELS = r"c:\Users\jayhi\Downloads\akshara-ocr\data\final\train\labels.txt"
VAL_LABELS = r"c:\Users\jayhi\Downloads\akshara-ocr\data\final\val\labels.txt"

def check_dataset(labels_file, name, check_n=100):
    print(f"\n=== Checking {name} ===")
    samples = []
    with open(labels_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                path, label = line.split('|', 1)
                samples.append((path.strip(), label.strip()))
    
    print(f"Total entries: {len(samples)}")
    
    failed = 0
    shapes = []
    for path, label in samples[:check_n]:
        try:
            img = Image.open(path).convert('L')
            shapes.append(img.size)
        except Exception as e:
            print(f"FAILED: {path} -> {e}")
            failed += 1
    
    print(f"Checked first {check_n} images: {check_n - failed} OK, {failed} FAILED")
    
    widths = [s[0] for s in shapes]
    print(f"Image widths: min={min(widths)}, max={max(widths)}, avg={int(sum(widths)/len(widths))}")
    print(f"Image height (all should be 40): {set(s[1] for s in shapes)}")
    
    # Show 5 samples
    print("5 sample entries:")
    for path, label in samples[10:15]:
        print(f"  {os.path.basename(path)} | {label}")

check_dataset(TRAIN_LABELS, "TRAIN")
check_dataset(VAL_LABELS, "VAL")
print("\n=== VOCAB CHECK ===")

# Collect all unique characters in labels
all_chars = set()
with open(TRAIN_LABELS, encoding='utf-8') as f:
    for line in f:
        if '|' in line:
            label = line.strip().split('|', 1)[1]
            all_chars.update(label)
print(f"Unique characters in training labels: {len(all_chars)}")
print(f"Characters: {''.join(sorted(all_chars))}")
