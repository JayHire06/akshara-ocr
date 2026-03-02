# data/synthetic_generator.py
import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from augmentation import apply_augmentations

random.seed(42)
np.random.seed(42)

def download_fonts(fonts_dir="fonts"):
    """
    Instructions to download Google Fonts: Noto Sans Devanagari, Noto Sans Tamil, Noto Sans Bengali, etc.
    Place the .ttf or .otf files in the fonts_dir.
    """
    os.makedirs(fonts_dir, exist_ok=True)
    print(f"Please ensure Google Fonts (like Noto Sans Devanagari) are downloaded to {fonts_dir}/ directory.")

def generate_sample(text, language_code, fonts_dir="fonts", output_dir="synthetic"):
    """Generates a synthetic image sample for the given text with augmentation."""
    os.makedirs(fonts_dir, exist_ok=True)
    font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf') or f.endswith('.otf')]
    if not font_files:
        raise FileNotFoundError(f"No fonts found in {fonts_dir}. Please add font files.")
        
    font_file = random.choice(font_files)
    font_path = os.path.join(fonts_dir, font_file)
    font_size = random.randint(18, 32)
    
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text bounding box to create appropriately sized image
    dummy_img = Image.new("L", (1, 1), "white")
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(bbox[2] - bbox[0] + 10, 10)
    height = max(bbox[3] - bbox[1] + 10, 10)
    
    img = Image.new("L", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, fill="black", font=font)
    
    # Apply augmentations
    img = apply_augmentations(img)
    
    lang_output_dir = os.path.join(output_dir, language_code)
    os.makedirs(lang_output_dir, exist_ok=True)
    
    filename = f"{language_code}_{random.randint(100000, 999999)}.png"
    filepath = os.path.join(lang_output_dir, filename)
    img.save(filepath)
    
    return filename, text, language_code, font_file, True

def generate_dataset(num_samples=100, output_dir="synthetic", fonts_dir="fonts", languages=None):
    all_texts = {
        "hi": ["नमस्ते", "भारत", "हिंदी", "देवनागरी", "अक्षर", "विज्ञान", "प्रौद्योगिकी", "ज्ञान", "कम्प्यूटर", "आर्टिफिशियल", "बुद्धिमत्ता", "मशीन", "डेटा", "सॉफ्टवेयर", "डेवलपर"],
        "ta": ["வணக்கம்", "இந்தியா", "தமிழ்"],
        "bn": ["নমস্কার", "ভারত", "বাংলা"],
        "gu": ["નમસ્તે", "ભારત", "ગુજરાતી"]
    }
    
    texts = {k: v for k, v in all_texts.items() if k in languages} if languages else all_texts
    
    os.makedirs(output_dir, exist_ok=True)
    labels_file = os.path.join(output_dir, "labels.csv")
    
    create_header = not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0
    
    with open(labels_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if create_header:
            writer.writerow(["filename", "text", "language", "font", "is_synthetic"])
            
        for _ in range(num_samples):
            lang = random.choice(list(texts.keys()))
            txt = random.choice(texts[lang])
            try:
                row = generate_sample(txt, lang, fonts_dir=fonts_dir, output_dir=output_dir)
                writer.writerow(row)
            except FileNotFoundError as e:
                print(e)
                break

if __name__ == '__main__':
    # Example usage
    download_fonts()
    print("Add valid fonts to the fonts/ directory, then run generate_dataset().")
    # generate_dataset(num_samples=10)
