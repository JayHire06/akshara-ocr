import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import io

random.seed(42)
np.random.seed(42)

def _find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    try:
        res = np.linalg.solve(A, B)
        return res.tolist()
    except np.linalg.LinAlgError:
        return [1, 0, 0, 0, 1, 0, 0, 0]

def apply_perspective(img, max_distortion=0.15):
    width, height = img.size
    
    xshift = int(width * max_distortion)
    yshift = int(height * max_distortion)
    
    nw = (random.randint(0, xshift), random.randint(0, yshift))
    ne = (width - random.randint(0, xshift), random.randint(0, yshift))
    sw = (random.randint(0, xshift), height - random.randint(0, yshift))
    se = (width - random.randint(0, xshift), height - random.randint(0, yshift))
    
    coeffs = _find_coeffs([(0, 0), (width, 0), (width, height), (0, height)], [nw, ne, se, sw])
    
    bg_color = img.getpixel((0,0)) if getattr(img, 'getpixel', None) else 255
    if isinstance(bg_color, tuple):
        bg_color = bg_color[0]
        
    return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=bg_color)

def apply_motion_blur(img, radius=2):
    arr = np.array(img).astype(np.float32)
    angle = random.uniform(0, 360)
    dx = np.cos(np.radians(angle))
    dy = np.sin(np.radians(angle))
    
    num_steps = radius * 2 + 1
    acc = np.zeros_like(arr)
    
    for i in range(num_steps):
        shift_x = int((i - radius) * dx)
        shift_y = int((i - radius) * dy)
        acc += np.roll(np.roll(arr, shift_x, axis=1), shift_y, axis=0)
        
    acc = acc / num_steps
    return Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))

def apply_shadow(img):
    width, height = img.size
    shadow = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(shadow)
    
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    start_intensity = random.randint(100, 200)
    end_intensity = 255
    
    if direction == 'horizontal':
        for x in range(width):
            intensity = int(start_intensity + (end_intensity - start_intensity) * (x / max(1, width)))
            draw.line([(x, 0), (x, height)], fill=intensity)
    elif direction == 'vertical':
        for y in range(height):
            intensity = int(start_intensity + (end_intensity - start_intensity) * (y / max(1, height)))
            draw.line([(0, y), (width, y)], fill=intensity)
    else: 
        for y in range(height):
            for x in range(width):
                factor = (x+y)/max(1, width+height)
                v = int(start_intensity + (end_intensity - start_intensity) * factor)
                shadow.putpixel((x,y), v)
                
    arr_img = np.array(img).astype(np.float32)
    arr_shadow = np.array(shadow).astype(np.float32) / 255.0
    arr = arr_img * arr_shadow
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_jpeg_artifacts(img, quality=None):
    if quality is None:
        quality = random.randint(15, 50)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")

def apply_scanner_lines(img):
    arr = np.array(img).astype(np.float32)
    num_lines = random.randint(1, 5)
    for _ in range(num_lines):
        line_intensity = random.randint(10, 50)
        if random.random() < 0.5:
            y = random.randint(0, max(0, img.size[1]-1))
            t = random.randint(1, 3)
            y_start = max(0, y-t)
            y_end = min(img.size[1], y+t)
            arr[y_start:y_end, :] -= line_intensity
        else:
            x = random.randint(0, max(0, img.size[0]-1))
            t = random.randint(1, 3)
            x_start = max(0, x-t)
            x_end = min(img.size[0], x+t)
            arr[:, x_start:x_end] -= line_intensity
            
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_ink_bleed(img):
    return img.filter(ImageFilter.MinFilter(3))

def apply_resolution_variation(img):
    factor = random.uniform(0.3, 0.7)
    orig_size = img.size
    small_size = (max(1, int(orig_size[0]*factor)), max(1, int(orig_size[1]*factor)))
    img = img.resize(small_size, Image.BILINEAR)
    img = img.resize(orig_size, Image.BICUBIC)
    return img

def add_paper_texture(img):
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(10, 30), arr.shape)
    arr = arr + noise
    
    x = np.linspace(0, 10, arr.shape[1])
    y = np.linspace(0, 10, arr.shape[0])
    xv, yv = np.meshgrid(x, y)
    structured_noise = np.sin(xv) * np.cos(yv) * random.uniform(5, 15)
    arr = arr + structured_noise
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_augmentations(img, severity="high"):
    if severity == "high":
        prob = 0.8
    else:
        prob = 0.3

    # Paper texture
    if random.random() < 0.8:
        img = add_paper_texture(img)

    # Perspective Transform (3D warp)
    if random.random() < prob:
        img = apply_perspective(img, max_distortion=random.uniform(0.02, 0.10))

    # Ink Bleed
    if random.random() < (prob * 0.5):
        img = apply_ink_bleed(img)

    # Blurs
    if random.random() < prob:
        if random.random() < 0.5:
            img = apply_motion_blur(img, radius=random.randint(1, 3))
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))

    # Shadows
    if random.random() < prob:
        img = apply_shadow(img)
        
    # Brightness shift
    enhancer = ImageEnhance.Brightness(img)
    if severity == "high":
        factor = random.uniform(0.5, 1.5)
    else:
        factor = random.uniform(0.8, 1.2)
    img = enhancer.enhance(factor)
        
    # Scanner lines
    if random.random() < (prob * 0.4):
        img = apply_scanner_lines(img)

    # Resolution variation
    if random.random() < prob:
        img = apply_resolution_variation(img)
        
    # JPEG Artifacts
    if random.random() < prob:
        if severity == "high":
            img = apply_jpeg_artifacts(img, quality=random.randint(15, 40))
        else:
            img = apply_jpeg_artifacts(img, quality=random.randint(40, 70))
            
    return img
