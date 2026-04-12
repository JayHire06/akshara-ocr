import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

font_dir = r'C:\Windows\Fonts'
hindi_fonts = []

for f in os.listdir(font_dir):
    if f.lower().endswith(('.ttf', '.otf')):
        path = os.path.join(font_dir, f)
        try:
            font = ImageFont.truetype(path, 24)
            test_img = Image.new('L', (100, 40), 255)
            draw = ImageDraw.Draw(test_img)
            draw.text((5, 5), 'क', font=font, fill=0)
            arr = np.array(test_img)
            if arr.min() < 200:
                hindi_fonts.append(path)
                print(f"HINDI OK: {f}")
            else:
                print(f"skip: {f}")
        except:
            pass

print(f"\nTotal Hindi-capable fonts: {len(hindi_fonts)}")
for f in hindi_fonts:
    print(f"  {f}")
