import pytest
import numpy as np
from PIL import Image
from preprocess.binarize import otsu_binarize

def test_otsu_binarize_bimodal(mock_image_bimodal):
    """Test Otsu binarization separates a clear bimodal image."""
    result = otsu_binarize(mock_image_bimodal)
    arr = np.array(result)
    
    # Assert values are only 0 and 255
    unique_vals = np.unique(arr)
    assert set(unique_vals).issubset({0, 255})
    
    # Check that the center region (originally 200) became 255
    assert arr[50, 50] == 255
    # The background with value 50 will be 0
    assert arr[10, 10] == 0

def test_otsu_binarize_blank(mock_image_blank):
    """Test Otsu binarization with a completely blank (white) image."""
    result = otsu_binarize(mock_image_blank)
    arr = np.array(result)
    assert np.all(arr == 0) or np.all(arr == 255)

def test_otsu_binarize_black(mock_image_black):
    """Test Otsu binarization with a completely black image."""
    result = otsu_binarize(mock_image_black)
    arr = np.array(result)
    assert np.all(arr == 0) or np.all(arr == 255)
