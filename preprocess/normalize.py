import numpy as np
import os
from PIL import Image

def normalize_line(line_image: Image.Image, target_height: int = 32) -> Image.Image:
    w, h = line_image.size
    if h == 0:
        h = 1
    aspect_ratio = w / h
    target_width = max(1, int(target_height * aspect_ratio))
    
    resized = line_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Normalize pixel values to [0, 1] float
    img_arr = np.array(resized, dtype=np.float32)
    normalized = img_arr / 255.0
    
    out_img = Image.fromarray(normalized, mode='F')
    
    if os.environ.get('DEBUG') == 'True':
        debug_arr = (normalized * 255).astype(np.uint8)
        debug_img = Image.fromarray(debug_arr, mode='L')
        debug_img.save('debug_normalize.png')
        
    return out_img
