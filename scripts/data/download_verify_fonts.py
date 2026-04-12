import urllib.request
import os

fonts_dir = r'c:\Users\jayhi\Downloads\akshara-ocr\data\fonts'
os.makedirs(fonts_dir, exist_ok=True)

# Google Fonts - free Devanagari fonts (direct download URLs)
fonts = {
    'Lohit-Devanagari.ttf': 'https://github.com/google/fonts/raw/main/ofl/tiro-devanagari-hindi/TiroDevanagariHindi-Regular.ttf',
    'Poppins-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf',
    'NotoSansDevanagari-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf',
    'NotoSerifDevanagari-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/notoserifdevanagari/NotoSerifDevanagari%5Bwdth%2Cwght%5D.ttf',
    'Mukta-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Regular.ttf',
    'Mukta-Bold.ttf': 'https://github.com/google/fonts/raw/main/ofl/mukta/Mukta-Bold.ttf',
    'Hind-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/hind/Hind-Regular.ttf',
    'Hind-Bold.ttf': 'https://github.com/google/fonts/raw/main/ofl/hind/Hind-Bold.ttf',
    'Baloo2-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/baloo2/Baloo2%5Bwght%5D.ttf',
    'DevLys010.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
}

for filename, url in fonts.items():
    out_path = os.path.join(fonts_dir, filename)
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, out_path)
        print(f"  OK: {out_path}")
    except Exception as e:
        print(f"  FAILED: {e}")

# Now verify which ones actually render Devanagari
print("\n=== VERIFYING DOWNLOADS ===")
from PIL import Image, ImageDraw, ImageFont
import numpy as np

confirmed = []
for filename in os.listdir(fonts_dir):
    if filename.endswith(('.ttf', '.otf')):
        path = os.path.join(fonts_dir, filename)
        try:
            font = ImageFont.truetype(path, 32)
            img = Image.new('RGB', (200, 60), (255,255,255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), 'नमस्ते', font=font, fill=(0,0,0))
            arr = np.array(img)
            dark_cols = (arr.min(axis=(0,2)) < 100).sum()
            if dark_cols > 30:
                confirmed.append(path)
                print(f"CONFIRMED: {filename} ({dark_cols} dark cols)")
            else:
                print(f"NO HINDI: {filename}")
        except Exception as e:
            print(f"ERROR {filename}: {e}")

print(f"\nTotal confirmed Hindi fonts: {len(confirmed)}")
