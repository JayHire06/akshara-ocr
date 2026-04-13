import numpy as np
from PIL import Image

from preprocess.binarize import local_mean_binarize, robust_binarize
from preprocess.pipeline import flatten_background


def test_local_mean_binarize_handles_gradient_background():
    arr = np.tile(np.linspace(120, 220, 120, dtype=np.uint8), (60, 1))
    arr[20:40, 35:85] = 40
    image = Image.fromarray(arr, mode='L')

    result = local_mean_binarize(image, window_size=21, offset=8)
    result_arr = np.array(result)

    assert result_arr[30, 60] == 0
    assert result_arr[5, 110] == 255


def test_robust_binarize_preserves_dark_text_on_bright_background():
    arr = np.ones((80, 160), dtype=np.uint8) * 235
    arr[:, :80] = 180
    arr[25:55, 50:120] = 30
    image = Image.fromarray(arr, mode='L')

    result = robust_binarize(image)
    result_arr = np.array(result)

    assert result_arr[40, 90] == 0
    assert result_arr[10, 10] == 255 or result_arr[10, 140] == 255


def test_flatten_background_reduces_large_scale_brightness_shift():
    arr = np.tile(np.linspace(80, 220, 120, dtype=np.uint8), (50, 1))
    arr[15:35, 45:90] = 30
    image = Image.fromarray(arr, mode='L')

    flattened = flatten_background(image)
    flat_arr = np.array(flattened, dtype=np.int16)

    left_bg = int(np.mean(flat_arr[:, 10:20]))
    right_bg = int(np.mean(flat_arr[:, 100:110]))
    assert abs(left_bg - right_bg) < 40
