import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image
import sys
import os
sys.path.insert(0, r'c:\Users\jayhi\Downloads\akshara-ocr')
from preprocess.segment import segment_page

VOCAB_FILE = r"c:\Users\jayhi\Downloads\akshara-ocr\data\vocab.json"
MODEL_PATH = r"c:\Users\jayhi\Downloads\akshara-ocr\model\checkpoints\best_model_v3.pth"

# Load vocab
with open(VOCAB_FILE, encoding='utf-8') as f:
    vocab = json.load(f)
vocab_inv = {v: k for k, v in vocab.items()}

# Model definition (same as train_v2.py)
class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2), stride=(2,1), padding=(0,1))
            )
        self.cnn = nn.Sequential(
            conv_block(1,64), conv_block(64,128), conv_block(128,256),
            conv_block(256,512), conv_block(512,512)
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, vocab_size)
    def forward(self, x):
        x = self.cnn(x).squeeze(2).permute(2,0,1)
        x, _ = self.rnn(x)
        return nn.functional.log_softmax(self.fc(x), dim=2)

def decode(log_probs):
    preds = torch.argmax(log_probs, dim=2).squeeze(1).tolist()
    chars, prev = [], -1
    for idx in preds:
        if idx != prev and idx != 0:
            chars.append(vocab_inv.get(idx, '?'))
        prev = idx
    return ''.join(chars)

def predict_word(word_img, model, device):
    """Takes a PIL word image, returns predicted string."""
    img = word_img.convert('L')
    w, h = img.size
    new_w = max(1, int(w * 32 / h))
    img = img.resize((new_w, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return decode(out)

def ocr_page(image_path):
    """Full pipeline: image path → extracted Hindi text."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(len(vocab)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Segment page into lines and words
    lines = segment_page(image_path)
    
    full_text = []
    for line_words in lines:
        line_text = []
        for word_img in line_words:
            pred = predict_word(word_img, model, device)
            if pred.strip():
                line_text.append(pred)
        if line_text:
            full_text.append(' '.join(line_text))
    
    return '\n'.join(full_text)

if __name__ == '__main__':
    # Test on real document images
    real_dir = r'c:\Users\jayhi\Downloads\akshara-ocr\data\real_test\hindi'
    
    print("=== FULL PAGE OCR TEST ===\n")
    for fname in sorted(os.listdir(real_dir)):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(real_dir, fname)
            print(f"--- {fname} ---")
            result = ocr_page(path)
            print(f"OUTPUT: {result}")
            print()
