import os
import random
import numpy as np
from PIL import Image

# Hardcoded paths
KAGGLE_PATH = r"c:\Users\jayhi\Downloads\Images\Images"
DAKSHINA_TRAIN = r"c:\Users\jayhi\Downloads\dakshina_dataset_v1.0\dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.train.tsv"
DAKSHINA_DEV = r"c:\Users\jayhi\Downloads\dakshina_dataset_v1.0\dakshina_dataset_v1.0\hi\lexicons\hi.translit.sampled.dev.tsv"
OUTPUT_TRAIN = r"c:\Users\jayhi\Downloads\akshara-ocr\data\final\train"
OUTPUT_VAL = r"c:\Users\jayhi\Downloads\akshara-ocr\data\final\val"

CHAR_MAP = {
    'character_01_ka': 'क', 'character_02_kha': 'ख', 'character_03_ga': 'ग',
    'character_04_gha': 'घ', 'character_05_kna': 'ङ', 'character_06_cha': 'च',
    'character_07_chha': 'छ', 'character_08_ja': 'ज', 'character_09_jha': 'झ',
    'character_10_yna': 'ञ', 'character_11_taamatar': 'ट', 'character_12_thaa': 'ठ',
    'character_13_daa': 'ड', 'character_14_dhaa': 'ढ', 'character_15_adna': 'ण',
    'character_16_tabala': 'त', 'character_17_tha': 'थ', 'character_18_da': 'द',
    'character_19_dha': 'ध', 'character_20_na': 'न', 'character_21_pa': 'प',
    'character_22_pha': 'फ', 'character_23_ba': 'ब', 'character_24_bha': 'भ',
    'character_25_ma': 'म', 'character_26_yaw': 'य', 'character_27_ra': 'र',
    'character_28_la': 'ल', 'character_29_waw': 'व', 'character_30_motosaw': 'श',
    'character_31_petchiryakha': 'ष', 'character_32_patalosaw': 'स', 'character_33_ha': 'ह',
    'character_34_chhya': 'क्ष', 'character_35_tra': 'त्र', 'character_36_gya': 'ज्ञ',
    'digit_0': '०', 'digit_1': '१', 'digit_2': '२', 'digit_3': '३', 'digit_4': '४',
    'digit_5': '५', 'digit_6': '६', 'digit_7': '७', 'digit_8': '८', 'digit_9': '९'
}

SKIP_CHARS = {'ा','ि','ी','ु','ू','े','ै','ो','ौ','ं','ः','्','ॅ','ॆ','ॉ','ॊ','़','ँ','ॄ'}

def load_character_bank(kaggle_path):
    """Loads all PNG images from each Kaggle folder into a dictionary mapping unicode_char -> list of PIL images."""
    char_bank = {v: [] for v in CHAR_MAP.values()}
    if not os.path.exists(kaggle_path):
        print(f"Warning: Kaggle dataset path not found: {kaggle_path}")
        return char_bank
        
    for folder_name in os.listdir(kaggle_path):
        if folder_name in CHAR_MAP:
            unicode_char = CHAR_MAP[folder_name]
            folder_path = os.path.join(kaggle_path, folder_name)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".png"):
                        filepath = os.path.join(folder_path, filename)
                        try:
                            # Load and keep image in memory (ensure it's not holding file handle)
                            with Image.open(filepath) as img:
                                char_bank[unicode_char].append(img.copy())
                        except Exception as e:
                            pass
    return char_bank

def load_dakshina_words(tsv_path):
    """Reads TSV, keeping column 0 words if ALL base characters are in CHAR_MAP, ignoring matras."""
    valid_words = []
    valid_chars = set(CHAR_MAP.values())
    
    if not os.path.exists(tsv_path):
        print(f"Warning: Dakshina TSV not found at {tsv_path}")
        return valid_words
        
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if parts:
                word = parts[0].strip()
                if word and all(char in valid_chars or char in SKIP_CHARS for char in word):
                    valid_words.append(word)
    return valid_words

def build_word_image(word, char_bank):
    """Composites individual character images into a single word image."""
    char_images = []
    
    # Check if all chars exist in bank AND we have images for them (skip matras)
    for char in word:
        if char in SKIP_CHARS:
            continue
            
        if char not in char_bank or not char_bank[char]:
            return None
            
        # Pick random image for character
        raw_img = random.choice(char_bank[char])
        
        # Resize to height=32px, keeping aspect ratio
        aspect_ratio = raw_img.width / raw_img.height
        new_width = int(32 * aspect_ratio)
        resized_img = raw_img.resize((max(1, new_width), 32), Image.BILINEAR).convert("L")
        char_images.append(resized_img)
        
    if not char_images:
        return None
        
    # Calculate total width: sum of char widths + (len-1)*4px gap
    total_width = sum(img.width for img in char_images) + (len(char_images) - 1) * 4
    
    # Create blank white canvas: height=40px, width=sum of all char widths + (len-1)*4px gap
    canvas = Image.new("L", (total_width, 40), 255)
    
    # Paste character images
    current_x = 0
    for img in char_images:
        # Center vertically: (40 - 32) // 2 = 4px padding top
        canvas.paste(img, (current_x, 4))
        current_x += img.width + 4
        
    # Apply numpy random gaussian noise (std=8)
    arr = np.array(canvas).astype(np.float32)
    noise = np.random.normal(0, 8, arr.shape)
    arr += noise
    
    # Clip values
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr)

def generate_split(words, output_dir, char_bank, variants_per_word):
    """Generates the datasets and labels.txt for a given split."""
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, "labels.txt")
    
    generated_count = 0
    samples_for_print = []
    
    with open(labels_file, "w", encoding="utf-8") as f:
        for word in words:
            for v in range(variants_per_word):
                img = build_word_image(word, char_bank)
                if img is not None:
                    # Save image
                    filename = f"{word}_{v}_{random.randint(10000, 99999)}.png"
                    filepath = os.path.join(images_dir, filename)
                    img.save(filepath)
                    
                    # Write label: absolute_image_path|hindi_word
                    abs_path = os.path.abspath(filepath)
                    stripped_label = ''.join(c for c in word if c not in SKIP_CHARS)
                    f.write(f"{abs_path}|{stripped_label}\n")
                    
                    if len(samples_for_print) < 5:
                        samples_for_print.append((abs_path, stripped_label))
                        
                    generated_count += 1
                    if generated_count % 1000 == 0:
                        print(f"Generated {generated_count} images for {output_dir}...")
                        
    return generated_count, samples_for_print

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("STEP 1: Loading character bank from Kaggle images...")
    char_bank = load_character_bank(KAGGLE_PATH)
    
    print("STEP 2: Loading valid Dakshina words...")
    train_words = load_dakshina_words(DAKSHINA_TRAIN)
    val_words = load_dakshina_words(DAKSHINA_DEV)
    print(f"Found {len(train_words)} valid train words and {len(val_words)} valid val words.")
    
    print("\nSTEP 3 & 4: Generating images for datasets...")
    print(f"Generating TRAIN set (5 variants per word)...")
    train_count, train_samples = generate_split(train_words, OUTPUT_TRAIN, char_bank, variants_per_word=5)
    
    print(f"\nGenerating VAL set (2 variants per word)...")
    val_count, val_samples = generate_split(val_words, OUTPUT_VAL, char_bank, variants_per_word=2)
    
    print("\n================ FINAL REPORT ================")
    print(f"Total TRAIN images generated: {train_count}")
    print(f"Total VAL images generated: {val_count}")
    
    print("\nSTEP 5: Sample predictions")
    for path, label in train_samples[:5]:
        print(f"SAMPLE: {path} | {label}")

if __name__ == '__main__':
    main()
