import os
import argparse
import json
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model.crnn import CRNN, CRNNv6
from model.crnn_v9 import CRNNv9

def export_to_onnx(model_path, vocab_path, output_path, model_version="v6"):
    """
    Exports the trained standard or edge models into a robust C++ ONNX graph mapping.
    """
    print(f"[*] Starting ONNX Export natively for {model_version}...")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    device = torch.device('cpu')

    if model_version == "v9":
        # v9 configs are read from the checkpoint's state-dict shape instead of
        # being hardcoded, so the same path exports both depth-3 and depth-6
        # variants correctly. EMA-overlaid weights land under the "model" key
        # in train_v9.save_checkpoint(); prefer that when present.
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
        layer_indices = {
            int(k.split(".")[2])
            for k in state_dict.keys()
            if k.startswith("transformer.layers.")
        }
        n_layers = (max(layer_indices) + 1) if layer_indices else 6
        model = CRNNv9(vocab_size, config={"transformer_layers": n_layers})
        print(f"[*] Detected CRNNv9 transformer_layers={n_layers}")
        model.load_state_dict(state_dict)
    else:
        # Legacy generic config for v1-v8 (LSTM-based)
        config = {
            "in_channels": 1,
            "lstm_hidden": 256,
            "lstm_layers": 2,
            "lstm_dropout": 0.0,
        }
        if model_version == "v6":
            model = CRNNv6(vocab_size, config)
        else:
            model = CRNN(vocab_size, config)
        print(f"[*] Loading state dictionary from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    
    # Use a generic dummy input
    dummy_input = torch.randn(1, 1, 32, 256)
    
    print(f"[*] Compiling ONNX Graph to {output_path}...")
    # Explicitly disable dynamo as it struggles with RNN flattening and Python 3.14 compatibility
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=20,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'width'},
            'output': {1: 'batch_size', 0: 'sequence_length'}
        },
        dynamo=False
    )
    
    print(f"[✓] Deployment Successful! Model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    print(f"[*] You can now move {output_path} to your frontend/public directory!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to best .pth model')
    parser.add_argument('--vocab', type=str, default='data/combined/vocab.json', help='Path to vocab.json')
    parser.add_argument('--out', type=str, default='frontend/public/model.onnx', help='Output ONNX path')
    parser.add_argument('--version', type=str, default='v9', choices=['legacy', 'v6', 'v9'], help='Model generation target')
    args = parser.parse_args()
    
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
    export_to_onnx(args.model, args.vocab, args.out, model_version=args.version)
