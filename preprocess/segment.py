import numpy as np
from PIL import Image, ImageEnhance

def binarize(img):
    """
    Apply Otsu thresholding by minimizing within-class variance.
    Input: PIL Image
    Returns: binary numpy array: 1 where text, 0 where background
    """
    img_gray = img.convert('L')
    img_arr = np.array(img_gray, dtype=np.uint8)
    
    hist = np.bincount(img_arr.ravel(), minlength=256)
    total_pixels = img_arr.size
    
    min_variance = float('inf')
    best_t = 0
    
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
        
        # calculate within-class variance
        # Var = sum((x - mean)^2 * freq) / weight
        
        # We can optimize this by calculating variance differently, but for the sake of 
        # direct formula translation given the size (256 bins), we can calculate it explicitly
        var_bg = sum(((i - mean_bg) ** 2) * hist[i] for i in range(0, t + 1)) / w_bg
        var_fg = sum(((i - mean_fg) ** 2) * hist[i] for i in range(t + 1, 256)) / w_fg
        
        # weighted sum of within-class variances
        within_class_variance = (w_bg / total_pixels) * var_bg + (w_fg / total_pixels) * var_fg
        
        if within_class_variance < min_variance:
            min_variance = within_class_variance
            best_t = t
            
    # Text is typically darker (values < best_t)
    binary_arr = np.where(img_arr <= best_t, 1, 0).astype(np.uint8)
    return binary_arr

def segment_lines(binary_arr):
    """
    Segments lines using horizontal projection.
    """
    # Compute horizontal projection: sum of each row
    proj = np.sum(binary_arr, axis=1)
    
    # Smooth the projection with a 5-pixel running average
    window = 5
    kernel = np.ones(window) / window
    proj_smooth = np.convolve(proj, kernel, mode='same')
    
    # Find line boundaries: a row is "text" if its smoothed sum > 2
    text_rows = proj_smooth > 2
    
    # Collect start/end row indices of consecutive text rows
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, is_text in enumerate(text_rows):
        if is_text and not in_segment:
            in_segment = True
            start_idx = i
        elif not is_text and in_segment:
            in_segment = False
            segments.append((start_idx, i))
            
    if in_segment:
        segments.append((start_idx, len(text_rows)))
        
    # Merge line segments that are less than 8px apart
    if not segments:
        return []
        
    merged_segments = [segments[0]]
    for current in segments[1:]:
        previous = merged_segments[-1]
        
        if current[0] - previous[1] < 8:
            # merge
            merged_segments[-1] = (previous[0], current[1])
        else:
            merged_segments.append(current)
            
    # Filter out segments shorter than 10px height
    final_segments = [s for s in merged_segments if (s[1] - s[0]) >= 10]
    
    return final_segments

def segment_words(line_binary_arr):
    """
    Segments words using vertical projection.
    """
    # Compute vertical projection: sum of each column
    proj = np.sum(line_binary_arr, axis=0)
    
    # Find word boundaries: a column is "text" if its sum > 0
    text_cols = proj > 0
    
    # Collect start/end column indices of consecutive text columns
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, is_text in enumerate(text_cols):
        if is_text and not in_segment:
            in_segment = True
            start_idx = i
        elif not is_text and in_segment:
            in_segment = False
            segments.append((start_idx, i))
            
    if in_segment:
        segments.append((start_idx, len(text_cols)))
        
    # Merge word segments that are less than 20px apart
    if not segments:
        return []
        
    merged_segments = [segments[0]]
    for current in segments[1:]:
        previous = merged_segments[-1]
        
        if current[0] - previous[1] < 20:
            # merge
            merged_segments[-1] = (previous[0], current[1])
        else:
            merged_segments.append(current)
            
    # Filter out segments narrower than 8px
    final_segments = [s for s in merged_segments if (s[1] - s[0]) >= 8]
    
    return final_segments

def segment_page(image_path):
    """
    Full page segmentation orchestration
    """
    # Load image
    original_img = Image.open(image_path)
    
    # Grayscale
    gray_img = original_img.convert('L')
    
    # Contrast Enhancement
    enhancer = ImageEnhance.Contrast(gray_img)
    enhanced_img = enhancer.enhance(2.0)
    
    # Binarize
    binary_arr = binarize(enhanced_img)
    
    # Remove horizontal lines
    # binarize() returns a 0/1 array. remove_horizontal_lines expects a PIL Image
    # Wait, binarize returns 1 where text, 0 where background!
    # Our remove_horizontal_lines expects 0 where text, 255 where background!
    # Let me translate binary_arr to 0/255 before passing 
    pil_binary = Image.fromarray(np.where(binary_arr == 1, 0, 255).astype(np.uint8), mode='L')
    
    # Import locally to avoid circular import since pipeline imports segment
    from .pipeline import remove_horizontal_lines
    cleaned_pil = remove_horizontal_lines(pil_binary)
    
    # Convert back to 0/1 binary_arr
    cleaned_arr = np.array(cleaned_pil)
    binary_arr = np.where(cleaned_arr == 0, 1, 0).astype(np.uint8)
    
    original_arr = np.array(gray_img)
    
    # Call segment_lines() to get line regions
    line_bounds = segment_lines(binary_arr)
    
    page_words = []
    
    # For each line:
    for r_start, r_end in line_bounds:
        line_binary = binary_arr[r_start:r_end, :]
        line_original = original_arr[r_start:r_end, :]
        
        # Call segment_words() on the line
        word_bounds = segment_words(line_binary)
        
        line_words = []
        for c_start, c_end in word_bounds:
            # Crop word
            word_arr = line_original[:, c_start:c_end]
            
            # Convert to PIL
            word_img = Image.fromarray(word_arr)
            
            # Resize to height=32px keeping aspect ratio
            w, h = word_img.size
            if h > 0:
                aspect_ratio = w / h
                new_width = max(1, int(32 * aspect_ratio))
                resized = word_img.resize((new_width, 32), Image.Resampling.LANCZOS)
                line_words.append(resized)
                
        page_words.append(line_words)
        
    # Return: list of lists
    return page_words


if __name__ == '__main__':
    import sys
    from PIL import Image
    
    # Create a simple test image with fake text blocks
    # to verify segmentation works without needing real images
    test_img = Image.new('L', (400, 200), color=255)
    import numpy as np
    arr = np.array(test_img)
    # Draw fake text blocks (black rectangles simulating words) < 40px wide so they aren't removed as lines
    arr[20:40, 10:35] = 0    # line 1, word 1
    arr[20:40, 60:90] = 0   # line 1, word 2 (gap 25)
    arr[20:40, 115:140] = 0  # line 1, word 3 (gap 25)
    arr[60:80, 10:35] = 0    # line 2, word 1
    arr[60:80, 60:90] = 0  # line 2, word 2 (gap 25)
    arr[100:120, 10:35] = 0 # line 3, word 1
    
    test_img = Image.fromarray(arr)
    test_img.save('test_page.png')
    
    result = segment_page('test_page.png')
    print(f"Lines found: {len(result)}")
    for i, line in enumerate(result):
        print(f"  Line {i}: {len(line)} words")
        for j, word in enumerate(line):
            print(f"    Word {j}: size={word.size}")
    
    expected_lines = 3
    expected_words = [3, 2, 1]
    print("\nExpected 3 lines with [3,2,1] words")
    print("PASS" if len(result) == 3 else "FAIL - wrong line count")
