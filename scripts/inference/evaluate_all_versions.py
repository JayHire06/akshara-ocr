#!/usr/bin/env python3
import os
import sys
import glob
import json
import torch
from PIL import Image
import numpy as np

# Adjust path to import codebase cleanly
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from model.crnn import CRNN
from model.ctc_decoder import CTCDecoder
from preprocess.pipeline import preprocess

# --- NATIVE METRIC CALCULATION (No external dependencies) ---
def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calculate_cer(pred, target):
    if len(target) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    return min(1.0, levenshtein_distance(pred, target) / len(target))

def calculate_wer(pred, target):
    pred_words = pred.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0
    return min(1.0, levenshtein_distance(pred_words, target_words) / len(target_words))

# --- MODEL INFERENCE WRAPPER ---
def evaluate_model_on_directory(model_identifier, checkpoint_path, real_test_dir, gt_file, device):
    print(f"Evaluating {model_identifier} -> {os.path.basename(checkpoint_path)}")
    
    # 1. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    if 'model' in ckpt: # v5 format
        state_dict = ckpt['model']
        
    # Infer Vocab Size Dynamically from Output FC layer (avoids tensor mismatches)
    fc_bias_key = [k for k in state_dict.keys() if 'fc.bias' in k or 'fc.2.bias' in k][-1]
    vocab_size = state_dict[fc_bias_key].shape[0]
    
    # 2. Build Model structure
    config = {
        'in_channels': 1, 'lstm_input_size': 512, 'lstm_hidden_size': 256,
        'lstm_num_layers': 2, 'lstm_dropout': 0.0, 'blank_id': 0
    }
    model = CRNN(vocab_size=vocab_size, config=config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. Mount appropriate vocab list structure
    # Fallback padding to ensure decoder array dimensions align perfectly with network topology runtime
    vocab_list = ["<PAD>"] + [chr(i) for i in range(33, 33 + vocab_size - 1)] 
    
    # Attempt to locate real vocab mappings used during execution
    standard_vocab = os.path.join(root_dir, 'vocab.json')
    if 'run_v5' in checkpoint_path:
        # V5 targets combined mapped vocab
        standard_vocab = os.path.join(root_dir, 'data', 'combined', 'vocab.json')
        
    if os.path.exists(standard_vocab):
        with open(standard_vocab, 'r', encoding='utf-8') as f:
            v_dict = json.load(f)
            # Reconstruct index map
            vocab_list = ["<UNK>"] * vocab_size
            for char, idx in v_dict.items():
                if idx < vocab_size:
                    if char == '<blank>':
                        vocab_list[idx] = "<PAD>"
                    else:
                        vocab_list[idx] = char

    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)

    # 4. Run Ground Truth Pipeline Evaluation
    total_cer, total_wer, total_samples = 0.0, 0.0, 0
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2: continue
        
        img_name, expected_text = parts[0], parts[1].strip()
        img_path = os.path.join(real_test_dir, img_name)
        
        if not os.path.exists(img_path): continue
        
        try:
            pil_img = Image.open(img_path).convert('RGB')
            # Natively pre-process (line segmentation wrapper for OCR)
            line_images = preprocess(pil_img, language='hi')
        except Exception:
            continue
            
        preds = []
        with torch.no_grad():
            for li in line_images:
                arr = np.array(li).astype(np.float32)
                if arr.max() > 1.0: arr = arr / 255.0
                
                arr_t = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 32, W)
                out = model(arr_t)
                decoded = decoder.decode_best_path(out)
                if decoded:
                    preds.append(decoded[0].replace('<UNK>', ''))
                    
        final_prediction_string = " ".join(preds).strip()
        
        # Calculate isolated distance rates
        total_cer += calculate_cer(final_prediction_string, expected_text)
        total_wer += calculate_wer(final_prediction_string, expected_text)
        total_samples += 1

    if total_samples == 0:
        return 0, 0

    return (total_cer / total_samples) * 100, (total_wer / total_samples) * 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("======================================================")
    print("   Akshara-OCR Cross-Version Architecture Benchmark   ")
    print("======================================================\n")

    real_test_dir = os.path.join(root_dir, 'data', 'real_test', 'hindi')
    gt_file = os.path.join(real_test_dir, 'ground_truth.txt')
    
    if not os.path.exists(gt_file):
        print(f"Ground truth file missing at: {gt_file}")
        print("Please ensure your real-world test sets form a ground_truth.txt mapping file.")
        return

    # Define Checkpoint target structures mapped chronologically
    models_to_test = {
        "v1 (Base Gen.)": os.path.join(root_dir, 'model', 'checkpoints_v1'),
        "v2 (Early Aug.)": os.path.join(root_dir, 'model', 'checkpoints_v2'),
        "v3 (200K Extended)": os.path.join(root_dir, 'model', 'checkpoints_200k'),
        "v4 (Realistic)": os.path.join(root_dir, 'model', 'checkpoints_realistic_v2'),
        "v5 (Current Prod)": os.path.join(root_dir, 'model', 'checkpoints')
    }
    
    results = []

    for name, directory in models_to_test.items():
        if not os.path.exists(directory):
            results.append((name, "N/A", "N/A"))
            continue
            
        # Dynamically mount the heaviest/best checkpoint locally in those maps
        ckpts = glob.glob(os.path.join(directory, "*.pth")) # Standard logic
        if name == "v5 (Current Prod)": # v5 logic has subfolders
             ckpts = glob.glob(os.path.join(directory, "run_v5_*", "best.pth"))
             
        if not ckpts:
            results.append((name, "N/A", "N/A"))
            continue
            
        # We usually prefer 'best_model', otherwise rely on latest timestamp modified tracking
        best_ckpt = max(ckpts, key=os.path.getmtime)
        try:
            cer, wer = evaluate_model_on_directory(name, best_ckpt, real_test_dir, gt_file, device)
            results.append((name, f"{cer:.2f}%", f"{wer:.2f}%"))
        except Exception as e:
            print(f"Failed to infer {name}: {e}")
            results.append((name, "Error", "Error"))

    # Table Generation Phase
    print("\n\n=============== PERFORMANCE REPORT ===============")
    print(f"{'Model Version':<25} | {'Val CER (Char Error)':<20} | {'WER (Word Error)':<20}")
    print("-" * 72)
    for res in results:
        print(f"{res[0]:<25} | {res[1]:<20} | {res[2]:<20}")
    print("==================================================")
    print("\nNote: Lower CER/WER string percentages explicitly indicate higher accuracy ratings against handwritten ground truths.")

if __name__ == "__main__":
    main()
