import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image
import os

VOCAB_FILE = r"c:\Users\jayhi\Downloads\akshara-ocr\data\vocab.json"
MODEL_PATH = r"c:\Users\jayhi\Downloads\akshara-ocr\model\checkpoints\best_model_v2.pth"

# Load vocab
with open(VOCAB_FILE, encoding='utf-8') as f:
    vocab = json.load(f)
vocab_inv = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# Load model (same architecture as train_v2.py)
class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,1), padding=(0,1))
            )
        self.cnn = nn.Sequential(
            conv_block(1, 64), conv_block(64, 128), conv_block(128, 256),
            conv_block(256, 512), conv_block(512, 512)
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3, batch_first=False)
        self.fc = nn.Linear(512, vocab_size)
    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(vocab_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

def decode(log_probs):
    preds = torch.argmax(log_probs, dim=2).squeeze(1).tolist()
    chars = []
    prev = -1
    for idx in preds:
        if idx != prev and idx != 0:
            chars.append(vocab_inv.get(idx, '?'))
        prev = idx
    return ''.join(chars)

def predict_image(path):
    img = Image.open(path).convert('L')
    w, h = img.size
    new_w = max(1, int(w * 32 / h))
    img = img.resize((new_w, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return decode(out)

# Test on 20 validation images from different parts of val set
val_labels = r"c:\Users\jayhi\Downloads\akshara-ocr\data\final\val\labels.txt"
with open(val_labels, encoding='utf-8') as f:
    lines = f.readlines()

print("=== VALIDATION SET SAMPLES (diverse) ===")
# Pick from different parts of val set
indices = [0, 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700,
           1900, 2100, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
correct = 0
for i in indices:
    if i < len(lines):
        path, label = lines[i].strip().split('|', 1)
        pred = predict_image(path)
        match = '✓' if pred == label else '✗'
        print(f"{match} GT: {label:20s} PRED: {pred}")
        if pred == label:
            correct += 1

print(f"\nAccuracy on sampled val images: {correct}/{len(indices)} = {correct/len(indices)*100:.1f}%")

# Now test on real Hindi document images if they exist
real_test_dir = r"c:\Users\jayhi\Downloads\akshara-ocr\data\real_test\hindi"
if os.path.exists(real_test_dir):
    print("\n=== REAL DOCUMENT IMAGES ===")
    for fname in os.listdir(real_test_dir)[:20]:
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(real_test_dir, fname)
            pred = predict_image(path)
            print(f"{fname}: {pred}")
else:
    print("\nNo real test images found at", real_test_dir)
    print("To test on real images, add Hindi word images to that folder")
