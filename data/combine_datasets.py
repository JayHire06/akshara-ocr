import os
import random

SOURCES = [
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\font_train\labels.txt',
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\font_augmented_train\labels.txt',
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\final\train\labels.txt',
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\iiit_real\labels.txt',
]
VAL_SOURCES = [
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\font_val\labels.txt',
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\final\val\labels.txt',
    r'c:\Users\jayhi\Downloads\akshara-ocr\data\iiit_real\val_labels.txt',
]
OUTPUT_TRAIN = r'c:\Users\jayhi\Downloads\akshara-ocr\data\combined\train_labels.txt'
OUTPUT_VAL = r'c:\Users\jayhi\Downloads\akshara-ocr\data\combined\val_labels.txt'

os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)

def process_sources(sources_list, output_path):
    all_lines = []
    source_counts = {}
    
    for filepath in sources_list:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            all_lines.extend(lines)
            source_counts[filepath] = len(lines)
            
    random.seed(42)
    random.shuffle(all_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in all_lines:
            f.write(f"{line}\n")
            
    return all_lines, source_counts

train_lines, train_counts = process_sources(SOURCES, OUTPUT_TRAIN)
val_lines, val_counts = process_sources(VAL_SOURCES, OUTPUT_VAL)

print(f"Total train count: {len(train_lines)}")
print(f"Total val count:   {len(val_lines)}")
print("\nBreakdown by source:")
print("TRAIN:")
for src, count in train_counts.items():
    print(f"  {os.path.basename(os.path.dirname(src))}: {count}")
print("VAL:")
for src, count in val_counts.items():
    print(f"  {os.path.basename(os.path.dirname(src))}: {count}")

print("\nSample lines from combined train:")
for line in train_lines[:5]:
    print(" ", line)
