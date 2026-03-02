import numpy as np
import os
from PIL import Image
from typing import List

def segment_lines(image: Image.Image) -> List[Image.Image]:
    img_arr = np.array(image)
    
    ink_arr = 255 - img_arr
    
    # 1. Compute horizontal projection: sum binary image along axis=1
    proj = np.sum(ink_arr, axis=1)
    
    # 2. Smooth the projection with a 1D moving average (window=5)
    window = 5
    kernel = np.ones(window) / window
    proj_smooth = np.convolve(proj, kernel, mode='same')
    
    # 3. Find rows where projection < threshold (these are line gaps)
    threshold = np.max(proj_smooth) * 0.05
    gaps = proj_smooth < threshold
    
    # 4. Split image at gap rows into individual line images
    lines = []
    in_line = False
    start_idx = 0
    
    for i, is_gap in enumerate(gaps):
        if not is_gap and not in_line:
            in_line = True
            start_idx = i
        elif is_gap and in_line:
            in_line = False
            end_idx = i
            line_arr = img_arr[start_idx:end_idx, :]
            lines.append(line_arr)
            
    if in_line:
        lines.append(img_arr[start_idx:, :])
        
    # 5. Filter out lines shorter than min_height (default 10px) or narrower than min_width (default 20px)
    min_height = 10
    min_width = 20
    
    valid_lines = []
    for line_arr in lines:
        if line_arr.shape[0] < min_height or line_arr.shape[1] < min_width:
            continue
        valid_lines.append(Image.fromarray(line_arr, mode=image.mode))
        
    if os.environ.get('DEBUG') == 'True':
        for idx, ln in enumerate(valid_lines):
            ln.save(f'debug_segment_line_{idx}.png')
            
    return valid_lines
