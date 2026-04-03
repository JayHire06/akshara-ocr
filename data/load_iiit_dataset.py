import os
import shutil

INPUT_BASE = r'C:\Users\jayhi\Downloads\hindi'
OUTPUT_BASE = r'c:\Users\jayhi\Downloads\akshara-ocr\data\iiit_real'

def process_split(split_name, output_labels_name):
    in_img_dir = os.path.join(INPUT_BASE, split_name, split_name, 'images')
    in_labels_txt = os.path.join(INPUT_BASE, split_name, split_name, f'{split_name}_labels.txt')
    in_images_txt = os.path.join(INPUT_BASE, split_name, split_name, f'{split_name}_images.txt')
    
    out_img_dir = os.path.join(OUTPUT_BASE, 'images')
    out_labels_txt = os.path.join(OUTPUT_BASE, output_labels_name)
    
    os.makedirs(out_img_dir, exist_ok=True)
    
    if not os.path.exists(in_labels_txt) or not os.path.exists(in_images_txt):
        print(f"Skipping {split_name}, files not found.")
        return 0
        
    with open(in_labels_txt, 'r', encoding='utf-8') as flabels, \
         open(in_images_txt, 'r', encoding='utf-8') as fimages:
        labels = flabels.read().splitlines()
        images = fimages.read().splitlines()
        
    if len(labels) != len(images):
        print(f"Length mismatch in {split_name}: {len(labels)} labels vs {len(images)} images.")
        return 0

    count = 0
    with open(out_labels_txt, 'w', encoding='utf-8') as fout:
        for lbl, img_name in zip(labels, images):
            in_img_path = os.path.join(in_img_dir, os.path.basename(img_name))
            if not os.path.exists(in_img_path):
                print(f"Missing image: {in_img_path}")
                continue
                
            safe_img_name = f"{split_name}_{os.path.basename(img_name)}"
            out_img_path = os.path.join(out_img_dir, safe_img_name)
            
            shutil.copy2(in_img_path, out_img_path)
            
            abs_path = os.path.abspath(out_img_path)
            fout.write(f"{abs_path}|{lbl}\n")
            count += 1
            
            if count % 10000 == 0:
                print(f"  ...copied {count} files for {split_name}")
            
    print(f"Finished copying {count} entries for {split_name}")
    return count

def main():
    print(f"Generating IIIT structure inside {OUTPUT_BASE}...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    train_count = process_split('train', 'labels.txt')
    val_count = process_split('val', 'val_labels.txt')
    
    print("\n=== SUMMARY ===")
    print(f"Total train labels written: {train_count}")
    print(f"Total val labels written: {val_count}")

if __name__ == '__main__':
    main()
