import os
from typing import List
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .binarize import robust_binarize
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


def flatten_background(image: Image.Image) -> Image.Image:
    """
    Reduce uneven illumination by subtracting a blurred background estimate.
    This is especially useful for phone photos with shadows or gradients.
    """
    gray = image.convert('L')
    blur_radius = max(6, min(gray.size) // 25) if gray.size else 6
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.array(gray, dtype=np.int16)
    bg = np.array(blurred, dtype=np.int16)
    flattened = np.clip(arr - bg + 128, 0, 255).astype(np.uint8)
    return Image.fromarray(flattened, mode='L')

def preprocess(image: Image.Image, language: str = 'hi') -> List[Image.Image]:
    """
    Complete custom image preprocessing pipeline.
    Input: a PIL Image (color, grayscale, or binary)
    Output: list of PIL Images, each is a binarized line image, height=32px, values [0,1] float
    """
    # 1. Convert colored images to grayscale
    gray_image = image.convert('L')

    # 2. Flatten lighting and then stretch contrast.
    flattened = flatten_background(gray_image)
    autocontrasted = ImageOps.autocontrast(flattened, cutoff=1)
    enhancer = ImageEnhance.Contrast(autocontrasted)
    enhanced_image = enhancer.enhance(1.8)

    # 3. Use robust binarization with local-threshold fallback.
    binary = robust_binarize(enhanced_image)

    # 4. Remove horizontal lines from ruled paper
    cleaned_binary = remove_horizontal_lines(binary)

    # 5. Continue pipeline
    deskewed = hough_deskew(cleaned_binary)
    denoised = denoise(deskewed)
    lines = segment_lines(denoised)
    
    processed_lines = []
    for line in lines:
        handled_line = handle_shirorekha(line, language)
        normalized = normalize_line(handled_line)
        processed_lines.append(normalized)
        
    return processed_lines
