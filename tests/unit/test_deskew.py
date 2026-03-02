import pytest
import numpy as np
from PIL import Image
from preprocess.deskew import hough_deskew
from unittest.mock import patch

def test_hough_deskew_straight_image():
    """Test deskew on an image that is already straight."""
    arr = np.ones((100, 100), dtype=np.uint8) * 255
    arr[45:55, 10:90] = 0  # Straight horizontal line
    img = Image.fromarray(arr, mode='L')
    
    with patch('PIL.Image.Image.rotate') as mock_rotate:
        mock_rotate.return_value = img
        result = hough_deskew(img)
        if mock_rotate.called:
            angle = mock_rotate.call_args[0][0]
            assert abs(angle) < 5.0  # Should be close to 0
        else:
            assert result is img

def test_hough_deskew_rotated_image():
    """Test deskew with an image containing a diagonal line."""
    arr = np.ones((100, 100), dtype=np.uint8) * 255
    for i in range(100):
        arr[i, i] = 0
        if i+1 < 100: arr[i+1, i] = 0
            
    img = Image.fromarray(arr, mode='L')
    
    with patch('PIL.Image.Image.rotate') as mock_rotate:
        mock_rotate.return_value = img
        hough_deskew(img)
        assert mock_rotate.called
        angle = mock_rotate.call_args[0][0]
        assert abs(angle) > 5.0  # Should detect a non-zero angle

def test_hough_deskew_blank_image(mock_image_blank):
    """Test deskew on blank image doesn't crash and returns original."""
    result = hough_deskew(mock_image_blank)
    assert result is mock_image_blank
