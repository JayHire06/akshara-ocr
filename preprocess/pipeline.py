import os
from typing import List
from PIL import Image, ImageEnhance
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .binarize import otsu_binarize
from .deskew import hough_deskew
from .denoise import denoise
from .segment import segment_lines
from .shirorekha import handle_shirorekha
from .normalize import normalize_line

def remove_horizontal_lines(image: Image.Image) -> Image.Image:
    """Removes horizontal lines from a binary image using morphological operations."""
    arr = np.array(image, dtype=np.uint8)
    
    # Text and lines are 0 (black), background is 255 (white)
    ink_mask = (arr == 0).astype(np.uint8)
    
    k_width = 40
    h, w = ink_mask.shape
    
    # Make kernel proportional to image width so short lines don't get
    # their character strokes erased. A ruled line spans the full width;
    # characters never do. Use 30% of image width, minimum 40px.
    k_width = max(40, int(w * 0.30))
    
    if w < k_width:
        return image
        
    pad_w = k_width // 2
    pad_right = k_width - 1 - pad_w
    
    # Erode horizontally (find min -> preserves long horizontal stretches of 1s)
    padded = np.pad(ink_mask, ((0, 0), (pad_w, pad_right)), mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (1, k_width))
    eroded = np.min(windows, axis=(2, 3)).reshape(h, w)
    
    # Dilate horizontally to restore original line thickness
    padded_for_dilate = np.pad(eroded, ((0, 0), (pad_w, pad_right)), mode='constant', constant_values=0)
    windows2 = sliding_window_view(padded_for_dilate, (1, k_width))
    dilated_lines = np.max(windows2, axis=(2, 3)).reshape(h, w)
    
    # Remove isolated lines from ink_mask (keep ink where there are no dilated lines isolated as 1s)
    clean_ink = ink_mask & (~dilated_lines)
    
    clean_arr = np.where(clean_ink, 0, 255).astype(np.uint8)
    
    out_img = Image.fromarray(clean_arr, mode='L')
    if os.environ.get('DEBUG') == 'True':
        out_img.save('debug_remove_lines.png')
        
    return out_img

def preprocess(image: Image.Image, language: str = 'hi') -> List[Image.Image]:
    """
    Complete custom image preprocessing pipeline.
    Input: a PIL Image (color, grayscale, or binary)
    Output: list of PIL Images, each is a binarized line image, height=32px, values [0,1] float
    """
    # 1. Convert colored images to grayscale before Otsu thresholding
    gray_image = image.convert('L')
    
    # 2. Add contrast enhancement using Pillow's ImageEnhance.Contrast
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)
    
    # 3. Apply Otsu Binarization
    binary = otsu_binarize(enhanced_image)
    
    # 4. Remove horizontal lines from lined paper
    cleaned_binary = remove_horizontal_lines(binary)
    
    # Continue pipeline
    deskewed = hough_deskew(cleaned_binary)
    denoised = denoise(deskewed)
    lines = segment_lines(denoised)
    
    processed_lines = []
    for line in lines:
        handled_line = handle_shirorekha(line, language)
        normalized = normalize_line(handled_line)
        processed_lines.append(normalized)
        
    return processed_lines
