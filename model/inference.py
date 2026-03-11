# FILE: /model/inference.py
import numpy as np
import onnxruntime as ort
from typing import List
from model.ctc_decoder import CTCDecoder
import json
import os

def recognize(line_images: List[np.ndarray]) -> List[str]:
    """
    Input: list of float32 arrays, width can be variable. shape of each: (1, 32, W)
    Output: list of raw decoded strings
    """
    model_path = os.path.join(os.path.dirname(__file__), 'checkpoints_200k', 'akshara_hindi_v1.onnx')
    
    # Needs to fall back gracefully if model missing, but assumes present.
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Warning: ONNX init failed. Error: {e}")
        return []
    
    # Vocab mock logic
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
            vocab_size = len(vocab_dict)
            vocab = [""] * vocab_size
            for char, idx in vocab_dict.items():
                vocab[idx] = char
    else:
        # Default ASCII range stub
        vocab = ['<PAD>'] + [chr(i) for i in range(32, 127)]
        
    decoder = CTCDecoder(vocab=vocab, blank_id=0)
    
    results = []
    for image in line_images:
        # Expected input shape constraint: (1, 1, 32, W)
        # Adding batch AND channel dimensions: (1, 1, 32, W)
        # Assuming original input image is (32, W), but interfaces says (1, 32, W).
        # Let's check shape and ensure it's 4D (1, 1, 32, W)
        if len(image.shape) == 2:
            input_data = np.expand_dims(image, axis=(0, 1)).astype(np.float32)
        elif len(image.shape) == 3:
            input_data = np.expand_dims(image, axis=0).astype(np.float32)
        else:
            input_data = image.astype(np.float32)
        
        ort_inputs = {session.get_inputs()[0].name: input_data}
        ort_outs = session.run(None, ort_inputs)
        
        # log_probs shape: (seq_len, batch_size=1, vocab_size)
        log_probs = ort_outs[0] 
        
        # We need a tensor for decoder. PyTorch is allowed as per constraint.
        import torch
        log_probs_tensor = torch.tensor(log_probs)
        decoded = decoder.decode_best_path(log_probs_tensor)
        
        if decoded:
            results.append(decoded[0])
        else:
            results.append("")
            
    return results
