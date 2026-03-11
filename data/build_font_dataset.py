import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS_DIR = r'c:\Users\jayhi\Downloads\akshara-ocr\data\fonts'
DAKSHINA_TRAIN = r'c:\Users\jayhi\Downloads\dakshina_dataset_v1.0\dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.train.tsv'
DAKSHINA_DEV = r'c:\Users\jayhi\Downloads\dakshina_dataset_v1.0\dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.dev.tsv'
OUTPUT_TRAIN = r'c:\Users\jayhi\Downloads\akshara-ocr\data\font_train'
OUTPUT_VAL = r'c:\Users\jayhi\Downloads\akshara-ocr\data\font_val'

print("STEP 1: Load all confirmed fonts")
confirmed_fonts = []
for filename in os.listdir(FONTS_DIR):
    if filename.endswith(('.ttf', '.otf')):
        path = os.path.join(FONTS_DIR, filename)
        try:
            font = ImageFont.truetype(path, 32)
            img = Image.new('RGB', (200, 60), color=(255,255,255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), 'नमस्ते', font=font, fill=(0,0,0))
            arr = np.array(img)
            dark_cols = (arr.min(axis=(0,2)) < 100).sum()
            if dark_cols > 30:
                confirmed_fonts.append((path, filename))
        except:
            pass
print(f"Total confirmed fonts loaded: {len(confirmed_fonts)}")

print("\nSTEP 2: Load words from Dakshina")
def load_words(tsv_path):
    words = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if parts:
                w = parts[0].strip()
                if w:
                    words.add(w)
    return list(words)

train_words = load_words(DAKSHINA_TRAIN)
val_words = load_words(DAKSHINA_DEV)
print(f"Train words deduplicated: {len(train_words)}")
print(f"Val words deduplicated: {len(val_words)}")

def render_word(word, font_path, font_size, noise_level):
    try:
        font = ImageFont.truetype(font_path, font_size)
        left, top, right, bottom = font.getbbox(word)
        width = right - left
        height = bottom - top
        
        # Add 4px padding on all sides
        canvas_width = width + 8
        canvas_height = height + 8
        
        if canvas_width <= 0 or canvas_height <= 0:
            return None
            
        img = Image.new('RGB', (canvas_width, canvas_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # We need to subtract the bounding box offsets to render correctly within our tight canvas
        x = 4 - left
        y = 4 - top
        draw.text((x, y), word, font=font, fill=(0,0,0))
        
        img = img.convert('L')
        # Resize to height=32px keeping aspect ratio
        aspect_ratio = img.width / img.height
        new_width = max(1, int(32 * aspect_ratio))
        img = img.resize((new_width, 32), Image.BILINEAR)
        
        arr = np.array(img).astype(np.float32) / 255.0
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, arr.shape)
            arr = arr + noise
            
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        
        return Image.fromarray(arr)
    except Exception as e:
        return None

print("\nSTEP 4 & 5: Generating datasets")

def generate_dataset(words, out_dir, variants_mode):
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    labels_file = os.path.join(out_dir, 'labels.txt')
    
    count = 0
    samples = []
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        for word in words:
            if variants_mode == 'train':
                fonts_to_use = random.sample(confirmed_fonts, min(3, len(confirmed_fonts)))
                for path, fname in fonts_to_use:
                    for size in [20, 28, 36]:
                        noise_level = random.choice([0.02, 0.05, 0.08])
                        img = render_word(word, path, size, noise_level)
                        if img:
                            img_name = f"{word}_{size}_{fname}_{random.randint(1000,9999)}.png"
                            img_path = os.path.join(out_dir, 'images', img_name)
                            img.save(img_path)
                            abs_path = os.path.abspath(img_path)
                            f.write(f"{abs_path}|{word}\n")
                            count += 1
                            if len(samples) < 5:
                                samples.append(f"{abs_path}|{word}")
                            if count % 5000 == 0:
                                print(f"  Generated {count} images...")
            elif variants_mode == 'val':
                fonts_to_use = random.sample(confirmed_fonts, min(2, len(confirmed_fonts)))
                for path, fname in fonts_to_use:
                    img = render_word(word, path, 28, 0.0)
                    if img:
                        img_name = f"{word}_val_{fname}_{random.randint(1000,9999)}.png"
                        img_path = os.path.join(out_dir, 'images', img_name)
                        img.save(img_path)
                        abs_path = os.path.abspath(img_path)
                        f.write(f"{abs_path}|{word}\n")
                        count += 1
                        if len(samples) < 5:
                            samples.append(f"{abs_path}|{word}")
                        if count % 5000 == 0:
                            print(f"  Generated {count} images...")
                            
    return count, samples

def main():
    random.seed(42)
    np.random.seed(42)

    train_count, train_samples = generate_dataset(train_words, OUTPUT_TRAIN, 'train')
    val_count, val_samples = generate_dataset(val_words, OUTPUT_VAL, 'val')

    print(f"\nTotal Train Output: {train_count}")
    print(f"Total Val Output: {val_count}")
    print("\nSample Train lines:")
    for s in train_samples:
        print(" ", s)
    print("\nSample Val lines:")
    for s in val_samples:
        print(" ", s)

if __name__ == '__main__':
    main()
