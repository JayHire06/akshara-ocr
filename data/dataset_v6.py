import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

class OCRDatasetV6(Dataset):
    """
    Advanced V6 Dataset employing OpenCV and Albumentations for robust spatial simulated degradation mapping.
    """
    def __init__(self, labels_file: Path, vocab: dict, is_training: bool = True):
        self.vocab = vocab
        self.is_training = is_training
        self.samples = []
        
        with open(labels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    path, label = line.split("|", 1)
                    if label:
                        self.samples.append((path.strip(), label.strip()))

        # Define physics augmentation pipeline based on empirical lens/ink artifacts
        if ALBUMENTATIONS_AVAILABLE and self.is_training:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=3, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.4),
                A.ElasticTransform(alpha=1, sigma=30, alpha_affine=20, p=0.3),
                A.GridDistortion(num_steps=4, distort_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CoarseDropout(max_holes=5, max_height=4, max_width=8, fill_value=255, p=0.3)
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Using OpenCV to pull image into numpy
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image at {path}")
            
            # Augment
            if self.transform is not None:
                augmented = self.transform(image=img)
                img = augmented['image']
                
            # PIL Style Resizing
            h, w = img.shape
            new_w = max(1, int(w * 32 / max(h, 1)))
            img_resized = cv2.resize(img, (new_w, 32), interpolation=cv2.INTER_LINEAR)
            
            # Normalize map
            arr = img_resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, 32, W)
            
        except Exception as e:
            # Fallback zero pad
            tensor = torch.zeros(1, 32, 64)

        encoded = [self.vocab[c] for c in label if c in self.vocab]
        return tensor, torch.tensor(encoded, dtype=torch.long), label
