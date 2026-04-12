from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# These are the only Windows fonts known to have real Devanagari support
candidates = [
    r'C:\Windows\Fonts\mangal.ttf',
    r'C:\Windows\Fonts\mangalb.ttf',
    r'C:\Windows\Fonts\Nirmala.ttf',
    r'C:\Windows\Fonts\NirmalaB.ttf',
    r'C:\Windows\Fonts\NirmalaS.ttf',
    r'C:\Windows\Fonts\kokila.ttf',
    r'C:\Windows\Fonts\kokilab.ttf',
    r'C:\Windows\Fonts\kokilai.ttf',
    r'C:\Windows\Fonts\kokilabi.ttf',
    r'C:\Windows\Fonts\utsaah.ttf',
    r'C:\Windows\Fonts\utsaahb.ttf',
    r'C:\Windows\Fonts\utsaahi.ttf',
    r'C:\Windows\Fonts\utsaahbi.ttf',
    r'C:\Windows\Fonts\aparaj.ttf',
    r'C:\Windows\Fonts\aparajb.ttf',
    r'C:\Windows\Fonts\aparaji.ttf',
    r'C:\Windows\Fonts\aparajbi.ttf',
    r'C:\Windows\Fonts\ARIALUNICODEMS.ttf',
    r'C:\Windows\Fonts\arialuni.ttf',
]

# Test word that uses multiple Devanagari-specific features
test_word = 'नमस्ते'

confirmed = []

for path in candidates:
    if not os.path.exists(path):
        print(f"NOT FOUND: {os.path.basename(path)}")
        continue
    try:
        font = ImageFont.truetype(path, 32)
        
        # Render test word
        img = Image.new('RGB', (200, 60), color=(255,255,255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), test_word, font=font, fill=(0,0,0))
        arr = np.array(img)
        
        # Check: rendered pixels should be dark (text) AND spread across width
        # A fallback box glyph would be a square - real Devanagari is wider
        dark_cols = (arr.min(axis=(0,2)) < 100).sum()
        
        if dark_cols > 30:  # real Devanagari word spans many columns
            confirmed.append(path)
            print(f"CONFIRMED DEVANAGARI: {os.path.basename(path)} ({dark_cols} dark cols)")
        else:
            print(f"LIKELY FALLBACK: {os.path.basename(path)} ({dark_cols} dark cols)")
    except Exception as e:
        print(f"ERROR {os.path.basename(path)}: {e}")

print(f"\nConfirmed Devanagari fonts: {len(confirmed)}")
for f in confirmed:
    print(f"  {f}")
