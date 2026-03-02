# data/augmentation.py
import numpy as np
from PIL import Image, ImageFilter
import random

random.seed(42)
np.random.seed(42)

def add_paper_texture(img):
    """Adds a subtle noise pattern simulating paper grain."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_salt_and_pepper_noise(img, amount=0.01):
    """Randomly sets amount fraction of pixels to 0 or 255."""
    arr = np.array(img)
    # Salt
    num_salt = np.ceil(amount * img.size[0] * img.size[1] / 2)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape[:2]]
    if len(coords) == 2 and len(coords[0]) > 0:
        arr[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(amount * img.size[0] * img.size[1] / 2)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape[:2]]
    if len(coords) == 2 and len(coords[0]) > 0:
        arr[tuple(coords)] = 0
    return Image.fromarray(arr)

def brightness_shift(img):
    """Multiply pixel values by random factor 0.7-1.3."""
    factor = random.uniform(0.7, 1.3)
    arr = np.array(img).astype(np.float32)
    arr = arr * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_augmentations(img):
    """Apply the full augmentation pipeline."""
    # Rotation: -5 to +5 degrees
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, fillcolor="white")
    
    # Gaussian blur: kernel size randomly 0, 1, or 3
    kernel_size = random.choice([0, 1, 3])
    if kernel_size > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=kernel_size))
        
    # Brightness shift
    img = brightness_shift(img)
    
    # Salt and pepper noise: 0-2%
    noise_amount = random.uniform(0, 0.02)
    if noise_amount > 0:
        img = add_salt_and_pepper_noise(img, amount=noise_amount)
        
    # Paper texture
    img = add_paper_texture(img)
    
    return img

if __name__ == '__main__':
    # Example usage
    example_img = Image.new("L", (100, 50), "white")
    aug_img = apply_augmentations(example_img)
    print("Augmentation applied successfully. Output size:", aug_img.size)
