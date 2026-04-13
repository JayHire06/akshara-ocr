import numpy as np
import os
from PIL import Image, ImageFilter


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


def local_mean_binarize(image: Image.Image, window_size: int = 31, offset: float = 10.0) -> Image.Image:
    img_gray = image.convert('L')
    img_arr = np.array(img_gray, dtype=np.uint8)
    if window_size % 2 == 0:
        window_size += 1

    radius = max(1, window_size // 2)
    mean_arr = np.array(img_gray.filter(ImageFilter.BoxBlur(radius=radius)), dtype=np.float32)
    global_mean = float(np.mean(img_arr))
    threshold = mean_arr - offset + np.maximum(0.0, (global_mean - mean_arr) * 0.5)
    binary_arr = np.where(img_arr > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_arr, mode='L')


def robust_binarize(image: Image.Image) -> Image.Image:
    """
    Prefer Otsu on clean scans, but fall back to local thresholding when the
    foreground ratio from Otsu looks unrealistic for document text.
    """
    otsu_img = otsu_binarize(image)
    otsu_arr = np.array(otsu_img, dtype=np.uint8)
    ink_ratio = float(np.mean(otsu_arr == 0))

    # Text on documents typically occupies a modest portion of the crop.
    # If Otsu produces an almost blank or almost full-ink page, local
    # thresholding is usually more stable under uneven lighting.
    if 0.01 <= ink_ratio <= 0.55:
        return otsu_img

    min_dim = min(image.size) if image.size else 32
    window = max(15, min(51, (min_dim // 6) | 1))
    return local_mean_binarize(image, window_size=window, offset=10.0)
