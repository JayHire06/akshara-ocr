import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from augmentation import apply_augmentations

random.seed(42)
np.random.seed(42)

def get_random_hindi_text():
    words = [
        "नमस्ते", "भारत", "हिंदी", "देवनागरी", "अक्षर", "विज्ञान", "प्रौद्योगिकी", 
        "ज्ञान", "कम्प्यूटर", "आर्टिफिशियल", "बुद्धिमत्ता", "मशीन", "डेटा", "सॉफ्टवेयर", "डेवलपर",
        "शिक्षा", "विद्यालय", "विश्वविद्यालय", "छात्र", "शिक्षक", "पुस्तक", "अध्ययन",
        "अनुसंधान", "तकनीक", "भविष्य", "नवाचार", "विकास", "सामाजिक", "आर्थिक", "प्रणाली"
    ]
    phrases = [
        "मेरा नाम क्या है", "भारत एक कृषी प्रधान देश है", "सत्यमेव जयते", "ज्ञान ही शक्ति है",
        "कठोर परिश्रम सफलता की कुंजी है", "समय धन से अधिक मूल्यवान है", "विद्या ददाति विनयम",
        "कला और संस्कृति", "आर्टिफिशियल इंटेलिजेंस का भविष्य", "डेटा विज्ञान के अनुप्रयोग"
    ]
    
    choice = random.random()
    if choice < 0.4:
        return random.choice(words)
    elif choice < 0.8:
        return " ".join(random.choices(words, k=random.randint(2, 4)))
    else:
        return random.choice(phrases)

def get_random_background(width, height):
    colors = [
        (255, 255, 255),  # White
        (255, 253, 208),  # Cream
        (245, 245, 220),  # Beige
        (220, 220, 220),  # Light gray
        (238, 232, 170),  # PaleGoldenrod
    ]
    base_color = random.choice(colors)
    img = Image.new("RGB", (width, height), base_color)
    
    if random.random() < 0.3:
        draw = ImageDraw.Draw(img)
        end_color = (max(0, base_color[0]-30), max(0, base_color[1]-30), max(0, base_color[2]-30))
        for y in range(height):
            r = int(base_color[0] + (end_color[0] - base_color[0]) * y / height)
            g = int(base_color[1] + (end_color[1] - base_color[1]) * y / height)
            b = int(base_color[2] + (end_color[2] - base_color[2]) * y / height)
            draw.line([(0, y), (width, y)], fill=(r,g,b))
            
    return img.convert("L")

def generate_sample(text, language_code, font_files, fonts_dir="fonts", output_dir="synthetic", split="train"):
    os.makedirs(fonts_dir, exist_ok=True)
    if not font_files:
        raise FileNotFoundError(f"No fonts found to generate samples.")
        
    font_file = random.choice(font_files)
    font_path = os.path.join(fonts_dir, font_file)
    font_size = random.randint(14, 40)
    
    font = ImageFont.truetype(font_path, font_size)
    
    # Calculate text bounding box
    dummy_img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(bbox[2] - bbox[0] + 10, 10)
    height = max(bbox[3] - bbox[1] + 10, 10)
    
    img = get_random_background(int(width), int(height))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), text, fill=random.randint(0, 50), font=font)
    
    # Variable character spacing via horizontal stretch
    if random.random() < 0.5:
        stretch_factor = random.uniform(0.9, 1.2)
        img = img.resize((int(img.width * stretch_factor), img.height), Image.BILINEAR)
    
    severity = "high" if split == "train" else "low"
    img = apply_augmentations(img, severity=severity)
    
    lang_output_dir = os.path.join(output_dir, split, language_code)
    os.makedirs(lang_output_dir, exist_ok=True)
    
    filename = f"{language_code}_{random.randint(1000000, 9999999)}.png"
    filepath = os.path.join(lang_output_dir, filename)
    img.save(filepath)
    
    return filename, text, language_code, font_file, True

def generate_dataset(num_samples=100, output_dir="synthetic", fonts_dir="fonts", languages=None):
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    files = {}
    
    for split in splits.keys():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        labels_file = os.path.join(split_dir, "labels.csv")
        create_header = not os.path.exists(labels_file) or os.stat(labels_file).st_size == 0
        f = open(labels_file, mode="a", newline="", encoding="utf-8")
        writer = csv.writer(f)
        if create_header:
            writer.writerow(["filename", "text", "language", "font", "is_synthetic"])
        files[split] = (f, writer)
        
    all_fonts = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf') or f.endswith('.otf')]
    if len(all_fonts) > 1:
        # Split fonts
        num_val = max(1, int(len(all_fonts) * 0.2))
        val_fonts = all_fonts[:num_val]
        train_fonts = all_fonts[num_val:]
    else:
        val_fonts = all_fonts
        train_fonts = all_fonts
        
    generated = 0
    for _ in range(num_samples):
        rand_val = random.random()
        if rand_val < splits["train"]:
            current_split = "train"
        elif rand_val < splits["train"] + splits["val"]:
            current_split = "val"
        else:
            current_split = "test"
            
        lang = random.choice(languages) if languages else "hi"
        txt = get_random_hindi_text() if lang == "hi" else "Text"
        
        fonts_to_use = val_fonts if current_split in ["val", "test"] else train_fonts
        
        try:
            row = generate_sample(txt, lang, font_files=fonts_to_use, fonts_dir=fonts_dir, output_dir=output_dir, split=current_split)
            files[current_split][1].writerow(row)
            generated += 1
            if generated % 10000 == 0:
                print(f"Progress: {generated}/{num_samples} images generated...")
        except FileNotFoundError as e:
            print(e)
            break
            
    for f, _ in files.values():
        f.close()
