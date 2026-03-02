import os
from typing import List
from PIL import Image

from .binarize import otsu_binarize
from .deskew import hough_deskew
from .denoise import denoise
from .segment import segment_lines
from .shirorekha import handle_shirorekha
from .normalize import normalize_line

def preprocess(image: Image.Image, language: str = 'hi') -> List[Image.Image]:
    """
    Complete custom image preprocessing pipeline.
    Input: a PIL Image (color, grayscale, or binary)
    Output: list of PIL Images, each is a binarized line image, height=32px, values [0,1] float
    """
    binary = otsu_binarize(image)
    deskewed = hough_deskew(binary)
    denoised = denoise(deskewed)
    lines = segment_lines(denoised)
    
    processed_lines = []
    for line in lines:
        handled_line = handle_shirorekha(line, language)
        normalized = normalize_line(handled_line)
        processed_lines.append(normalized)
        
    return processed_lines
