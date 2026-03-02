import numpy as np
import os
from PIL import Image

def handle_shirorekha(line_image: Image.Image, language: str) -> Image.Image:
    if language != 'hi':
        return line_image
        
    img_arr = np.array(line_image)
    ink_arr = 255 - img_arr
    h, w = ink_arr.shape
    
    # 1. For each line image in Devanagari, find the row with maximum ink density in the top 30% of the image
    top_30_idx = max(1, int(h * 0.30))
    proj = np.sum(ink_arr[:top_30_idx, :], axis=1)
    
    # 2. That row is the Shirorekha
    if len(proj) > 0:
        shirorekha_row = np.argmax(proj)
        
        # 3. Do NOT use it as a segmentation boundary — the characters connect through it
        if os.environ.get('DEBUG') == 'True':
            debug_arr = np.copy(img_arr)
            debug_arr[shirorekha_row, :] = 128
            out_img = Image.fromarray(debug_arr, mode=line_image.mode)
            out_img.save('debug_shirorekha.png')
            
    return line_image
