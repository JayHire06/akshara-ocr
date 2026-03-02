import pytest
import numpy as np
from PIL import Image
from preprocess.segment import segment_lines

def test_segment_lines_with_text(mock_image_text_lines):
    """Test line segmentation on an image with clear horizontal gaps."""
    lines = segment_lines(mock_image_text_lines)
    
    # Expect 3 lines to be extracted based on fixture
    assert len(lines) == 3
    
    for line in lines:
        arr = np.array(line)
        assert arr.shape[0] >= 10
        assert arr.shape[1] >= 20

def test_segment_lines_blank(mock_image_blank):
    """Test line segmentation on a blank image returns no valid lines."""
    lines = segment_lines(mock_image_blank)
    assert len(lines) == 0

def test_segment_lines_too_small():
    """Test segmentation ignores lines that are too small."""
    arr = np.ones((100, 100), dtype=np.uint8) * 255
    # Tiny line 5x10 (skipped, height < 10)
    arr[10:15, 10:20] = 0
    # Valid line 20x30
    arr[30:50, 10:40] = 0
    
    img = Image.fromarray(arr, mode='L')
    lines = segment_lines(img)
    
    assert len(lines) == 1
    assert np.array(lines[0]).shape[0] >= 20
