import numpy as np
import os
from PIL import Image

def hough_deskew(image: Image.Image) -> Image.Image:
    img_gray = image.convert('L')
    img_arr = np.array(img_gray, dtype=np.float32)
    
    # 1. Apply edge detection: simple Sobel-like gradient using NumPy convolution
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    padded = np.pad(img_arr, 1, mode='edge')
    gx = np.zeros_like(img_arr)
    gy = np.zeros_like(img_arr)
    
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * padded[i:i+img_arr.shape[0], j:j+img_arr.shape[1]]
            gy += ky[i, j] * padded[i:i+img_arr.shape[0], j:j+img_arr.shape[1]]
            
    magnitude = np.sqrt(gx**2 + gy**2)
    edges = magnitude > np.percentile(magnitude, 95)
    
    y_idxs, x_idxs = np.nonzero(edges)
    
    if len(x_idxs) > 20000:
        step = max(1, len(x_idxs) // 20000)
        x_idxs = x_idxs[::step][:20000]
        y_idxs = y_idxs[::step][:20000]
        
    # 2. Compute the Hough accumulator for lines (theta, rho)
    thetas = np.deg2rad(np.arange(-45, 45, 0.5))
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    
    h, w = img_arr.shape
    diag_len = int(np.ceil(np.sqrt(h**2 + w**2)))
    num_rhos = 2 * diag_len + 1
    accumulator = np.zeros((num_rhos, len(thetas)), dtype=np.int32)
    
    for i in range(len(thetas)):
        rhos = np.round(-x_idxs * sin_t[i] + y_idxs * cos_t[i]).astype(np.int32) + diag_len
        valid_rhos = rhos[(rhos >= 0) & (rhos < num_rhos)]
        counts = np.bincount(valid_rhos, minlength=num_rhos)
        accumulator[:, i] += counts[:num_rhos]
        
    # 3. Find the dominant theta (angle of most text lines)
    if accumulator.sum() == 0:
        return image
        
    idx = np.argmax(accumulator)
    rho_idx, theta_idx = np.unravel_index(idx, accumulator.shape)
    theta = np.rad2deg(thetas[theta_idx])
    
    # 4. Rotate the original image by -theta using PIL Image.rotate
    # Ignore unrealistic large corrections; text-line skew should be modest.
    theta = float(np.clip(theta, -15.0, 15.0))
    rotated_img = image.rotate(-theta, expand=True, fillcolor=255)
    
    if os.environ.get('DEBUG') == 'True':
        rotated_img.save('debug_deskewed.png')
        
    return rotated_img
