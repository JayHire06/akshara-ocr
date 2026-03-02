import numpy as np
import os
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

def erode(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological erosion in NumPy."""
    pad_w = kernel_size // 2
    padded = np.pad(img_arr, pad_w, mode='edge')
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return np.min(windows, axis=(2, 3))

def dilate(img_arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological dilation in NumPy."""
    pad_w = kernel_size // 2
    padded = np.pad(img_arr, pad_w, mode='edge')
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return np.max(windows, axis=(2, 3))

def denoise(image: Image.Image, kernel_size: int = 3) -> Image.Image:
    """Denoise using morphological erosion and dilation in NumPy."""
    img_arr = np.array(image, dtype=np.uint8)
    
    dilated = dilate(img_arr, kernel_size)
    eroded = erode(dilated, kernel_size)
    
    out_img = Image.fromarray(eroded.astype(np.uint8), mode=image.mode)
    
    if os.environ.get('DEBUG') == 'True':
        out_img.save('debug_denoise.png')
        
    return out_img
