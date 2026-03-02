import numpy as np
import os
from PIL import Image

def otsu_binarize(image: Image.Image) -> Image.Image:
    # 1. Convert image to grayscale NumPy array (uint8)
    img_gray = image.convert('L')
    img_arr = np.array(img_gray, dtype=np.uint8)
    
    # 2. Compute 256-bin histogram using np.bincount()
    hist = np.bincount(img_arr.ravel(), minlength=256)
    
    # 3. For each threshold t in 0..255, compute between-class variance: w0*w1*(mean0-mean1)^2
    total_pixels = img_arr.size
    best_t = 0
    max_variance = -1.0
    
    sum_all = sum(i * h for i, h in enumerate(hist))
    sum_bg = 0
    w_bg = 0
    
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
            
        w_fg = total_pixels - w_bg
        if w_fg == 0:
            break
            
        sum_bg += t * hist[t]
        
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        
        # between-class variance
        variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        
        # 4. Select t that maximizes between-class variance
        if variance > max_variance:
            max_variance = variance
            best_t = t
            
    # 5. Return binary image: pixels > t -> 255, else -> 0
    binary_arr = np.where(img_arr > best_t, 255, 0).astype(np.uint8)
    out_img = Image.fromarray(binary_arr, mode='L')
    
    if os.environ.get('DEBUG') == 'True':
        out_img.save('debug_otsu_binarize.png')
        
    return out_img
