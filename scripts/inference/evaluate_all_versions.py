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

from model.crnn import CRNN, CRNNv6
from model.crnn_v9 import CRNNv9
from model.ctc_decoder import CTCDecoder
from model.benchmark_eval import CheckpointLoadError
from preprocess.pipeline import preprocess

import unicodedata

# --- NATIVE METRIC CALCULATION ---
def normalize_text(text):
    if not text: return ""
    return unicodedata.normalize('NFC', text).strip()

def levenshtein_distance(s1, s2):
    # Ensure we work on characters, not bytes
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
    pred, target = normalize_text(pred), normalize_text(target)
    if not target:
        return 1.0 if pred else 0.0
    dist = levenshtein_distance(pred, target)
    return min(1.0, dist / len(target))

def calculate_wer(pred, target):
    pred, target = normalize_text(pred), normalize_text(target)
    pred_words = pred.split()
    target_words = target.split()
    if not target_words:
        return 1.0 if pred_words else 0.0
    dist = levenshtein_distance(pred_words, target_words)
    return min(1.0, dist / len(target_words))

# --- MODEL INFERENCE WRAPPER ---
def evaluate_model_on_directory(model_identifier, checkpoint_path, test_dir, gt_file, device):
    # 1. Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    if 'model' in ckpt:
        state_dict = ckpt['model']

    # Infer Vocab Size Dynamically
    fc_bias_key = [k for k in state_dict.keys() if '.fc.bias' in k or '.fc.2.bias' in k or 'linear.bias' in k or k == 'fc.bias']
    if not fc_bias_key:
        raise CheckpointLoadError(f"{checkpoint_path}: no fc/linear output bias key found.")
    fc_bias_key = fc_bias_key[-1]
    vocab_size = state_dict[fc_bias_key].shape[0]
    print(f"[{model_identifier}] Detected Vocab Size: {vocab_size}")

    # 2. Build Model structure based on dynamic key detection
    has_stn = any('stn.' in k for k in state_dict.keys())
    has_depthwise = any('depthwise' in k for k in state_dict.keys())
    has_block_cnn = any(k.startswith('cnn.block') for k in state_dict.keys())
    has_transformer = any('transformer.layers' in k for k in state_dict.keys())
    has_sequential_cnn = (
        any(k.startswith('cnn.0.0') for k in state_dict.keys()) and not has_depthwise
    )

    if has_sequential_cnn:
        raise CheckpointLoadError(
            f"{checkpoint_path}: legacy Sequential CRNN layout detected "
            f"(cnn.0.0.*/fc.*). Current CRNN uses cnn.block*/linear.*. "
            f"Retrain required."
        )

    config = {
        'in_channels': 1, 'lstm_input_size': 512, 'lstm_hidden_size': 256,
        'lstm_num_layers': 2, 'lstm_dropout': 0.0, 'blank_id': 0,
        'use_stn': has_stn,
        'lstm_hidden': 256, 'lstm_layers': 2,
    }

    if has_transformer:
        print(f"  [ARCH] Detected CRNNv9 (Transformer encoder)")
        model = CRNNv9(vocab_size=vocab_size, config=config).to(device)
        arch_name = "CRNNv9"
    elif has_depthwise or has_stn:
        print(f"  [ARCH] Detected CRNNv6 (MobileNet) | STN: {has_stn}")
        model = CRNNv6(vocab_size=vocab_size, config=config).to(device)
        arch_name = "CRNNv6"
    else:
        if not has_block_cnn:
            raise CheckpointLoadError(
                f"{checkpoint_path}: unrecognised CRNN variant."
            )
        print(f"  [ARCH] Detected Standard CRNN (blocks)")
        model = CRNN(vocab_size=vocab_size, config=config).to(device)
        arch_name = "CRNN"

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected = len(model.state_dict())
    loaded_ratio = (expected - len(missing)) / expected
    if loaded_ratio < 0.90:
        raise CheckpointLoadError(
            f"{checkpoint_path}: only {loaded_ratio*100:.0f}% of {arch_name} "
            f"params loaded (missing={len(missing)} unexpected={len(unexpected)}). "
            f"Refusing to report metrics on a random-init model."
        )
    model.eval()

    # 3. Handle Vocab
    vocab_list = ["<PAD>"] + [chr(i) for i in range(33, 33 + vocab_size - 1)] 
    
    # 3. Handle Vocab - Prioritize data/combined/vocab.json for consistent mapping
    possible_vocabs = [
        os.path.join(root_dir, 'data', 'combined', 'vocab.json'),
        os.path.join(root_dir, 'vocab.json')
    ]
    
    vocab_found = False
    for v_path in possible_vocabs:
        if os.path.exists(v_path):
            with open(v_path, 'r', encoding='utf-8') as f:
                v_dict = json.load(f)
                vocab_list = ["<UNK>"] * vocab_size
                for char, idx in v_dict.items():
                    if idx < vocab_size:
                        if char == '<blank>': vocab_list[idx] = "<PAD>"
                        else: vocab_list[idx] = char
                vocab_found = True
            break
            
    if not vocab_found:
        print(f"  [WARN] No vocab.json found, falling back to ASCII-range placeholder")
        vocab_list = ["<PAD>"] + [chr(i) for i in range(33, 33 + vocab_size - 1)]

    decoder = CTCDecoder(vocab=vocab_list, blank_id=0)

    # 4. Run Evaluation
    total_cer, total_wer, total_samples, empty_preds = 0.0, 0.0, 0, 0
    verbose_samples = 3
    
    print(f"[{model_identifier}] Model inference in progress...")
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for idx, line in enumerate(lines):
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        
        img_path, expected_text = parts[0], parts[1].strip()
        if not os.path.exists(img_path): continue
        
        try:
            pil_img = Image.open(img_path).convert('L')
            w, h = pil_img.size
            pil_img = pil_img.resize((int(w * 32 / h), 32))
            
            arr = np.array(pil_img).astype(np.float32) / 255.0
            arr_t = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(arr_t)
                decoded = decoder.decode_best_path(out)
                
            final_prediction = ""
            if decoded:
                final_prediction = decoded[0].replace('<UNK>', '').strip()
            
            if not final_prediction:
                empty_preds += 1
            
            # Normalization is now inside calculate functions
            total_cer += calculate_cer(final_prediction, expected_text)
            total_wer += calculate_wer(final_prediction, expected_text)
            total_samples += 1
            
            if idx < verbose_samples:
                pn = normalize_text(final_prediction)
                tn = normalize_text(expected_text)
                print(f"  Sample {idx+1}:")
                print(f"    GT: {repr(tn)}")
                print(f"    PR: {repr(pn)}")
                print(f"    Match: {pn == tn}")
            
        except Exception as e:
            continue

    if total_samples == 0: return 0, 0, 0
    return (total_cer / total_samples) * 100, (total_wer / total_samples) * 100, (empty_preds / total_samples) * 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("======================================================")
    print("   Akshara-OCR Cross-Version Architecture Benchmark   ")
    print("======================================================\n")

    test_dir = os.path.join(root_dir, 'data', 'combined', 'val')
    gt_file = os.path.join(root_dir, 'data', 'combined', 'val_labels.txt')
    
    if not os.path.exists(gt_file):
        print(f"Ground truth file missing at: {gt_file}")
        return

    models_to_test = {
        "v1 (Base Gen.)": os.path.join(root_dir, 'model', 'checkpoints_v1'),
        "v2 (Early Aug.)": os.path.join(root_dir, 'model', 'checkpoints_v2'),
        "v3 (200K Extended)": os.path.join(root_dir, 'model', 'checkpoints_200k'),
        "v4 (Realistic)": os.path.join(root_dir, 'model', 'checkpoints_realistic_v2'),
        "v5 (Current Prod)": os.path.join(root_dir, 'model', 'checkpoints'),
        "v6 (Edge STN)": os.path.join(root_dir, 'model', 'checkpoints_v6_edge'),
        "v7 (NLP-Guided)": os.path.join(root_dir, 'model', 'checkpoints_v7_nlp'),
        "v8 (Staged Curric.)": os.path.join(root_dir, 'model', 'checkpoints_v8'),
        "v9 (Transformer)":    os.path.join(root_dir, 'model', 'checkpoints_v9')
    }
    
    results = []

    for name, directory in models_to_test.items():
        if not os.path.exists(directory):
            results.append((name, "N/A", "N/A"))
            continue
            
        ckpts = glob.glob(os.path.join(directory, "*.pth"))
        if name == "v5 (Current Prod)":
             ckpts = glob.glob(os.path.join(directory, "run_v5_*", "best.pth"))
        elif name == "v6 (Edge STN)":
             ckpts = glob.glob(os.path.join(directory, "run_v6_*", "best_v6.pth"))
        elif name == "v7 (NLP-Guided)":
             ckpts = glob.glob(os.path.join(directory, "run_v7_*", "best_v7.pth"))
        elif name == "v8 (Staged Curric.)":
             ckpts = glob.glob(os.path.join(directory, "run_v8_*", "best_v8.pth"))
        elif name == "v9 (Transformer)":
             ckpts = glob.glob(os.path.join(directory, "run_v9_*", "best_v9.pth"))

        if not ckpts:
            results.append((name, "N/A", "N/A"))
            continue
            
        best_ckpt = max(ckpts, key=os.path.getmtime)
        try:
            cer, wer, empty = evaluate_model_on_directory(name, best_ckpt, test_dir, gt_file, device)
            results.append((name, f"{cer:.2f}%", f"{wer:.2f}%", f"{empty:.2f}%"))
        except CheckpointLoadError as err:
            print(f"[{name}] INCOMPATIBLE: {err}")
            results.append((name, "INCOMPAT", "INCOMPAT", "INCOMPAT"))
        except Exception as err:
            print(f"Failed to infer {name}: {err}")
            import traceback
            traceback.print_exc()
            results.append((name, "Error", "Error", "Error"))

    print("\n\n=============== PERFORMANCE REPORT ===============")
    print(f"{'Model Version':<25} | {'Val CER':<10} | {'WER':<10} | {'Empty Rate':<10}")
    print("-" * 75)
    for res in results:
        print(f"{res[0]:<25} | {res[1]:<10} | {res[2]:<10} | {res[3]:<10}")
    print("==================================================")

if __name__ == "__main__":
    main()
