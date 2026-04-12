import os
import argparse
import json
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.crnn import CRNN, CRNNv6

def export_to_onnx(model_path, vocab_path, output_path, model_version="v6"):
    """
    Exports the trained standard or edge models into a robust C++ ONNX graph mapping.
    """
    print(f"[*] Starting ONNX Export natively for {model_version}...")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    device = torch.device('cpu')
    
    # Generic Config Mappings
    config = {
        "in_channels": 1,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "lstm_dropout": 0.0
    }
    
    if model_version == "v6":
        model = CRNNv6(vocab_size, config)
    else:
        model = CRNN(vocab_size, config)
        
    print(f"[*] Loading state dictionary from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy input (Batch, Channels, Height, Width) -> Width is dynamic
    dummy_input = torch.randn(1, 1, 32, 128)
    
    # Define dynamic axes natively so the browser can pass heavily variable width strings
    dynamic_axes = {
        'input':  {0: 'batch_size', 3: 'width'},
        'output': {1: 'batch_size', 0: 'sequence_length'}
    }
    
    print(f"[*] Compiling ONNX Graph to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"[✓] Deployment Successful! Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"[*] You can now move {output_path} to your frontend/public directory!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to best .pth model')
    parser.add_argument('--vocab', type=str, default='data/combined/vocab.json', help='Path to vocab.json')
    parser.add_argument('--out', type=str, default='frontend/public/model.onnx', help='Output ONNX path')
    parser.add_argument('--version', type=str, default='v6', choices=['legacy', 'v6'], help='Model generation target')
    args = parser.parse_args()
    
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
    export_to_onnx(args.model, args.vocab, args.out, model_version=args.version)
