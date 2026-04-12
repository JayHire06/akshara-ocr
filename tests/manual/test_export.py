import torch
import torch.nn as nn
from model.crnn import CRNN

model = CRNN(vocab_size=96, config={})
model.eval()
dummy_input = torch.randn(1, 1, 32, 128)

try:
    torch.onnx.export(
        model, dummy_input, "test.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {3: 'width'}, 'output': {0: 'seq_len'}},
        fallback=True
    )
    print("Export Success with fallback=True")
except Exception as e:
    print(f"Fallback Failed: {e}")

