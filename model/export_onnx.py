# FILE: /model/export_onnx.py
import torch
from model.crnn import CRNN
import sys

def export_to_onnx(checkpoint_path: str, output_path: str, vocab_size: int, config: dict):
    model = CRNN(vocab_size=vocab_size, config=config)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Input: grayscale line image, height=32, width=variable
    dummy_input = torch.randn(1, 1, 32, 128)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {3: 'width'}, 
            'output': {0: 'seq_len'}
        },
        dynamo=False
    )
    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 3:
        chkpt = sys.argv[1]
        out_onnx = sys.argv[2]
        v_size = int(sys.argv[3])
        export_to_onnx(chkpt, out_onnx, vocab_size=v_size, config={})
