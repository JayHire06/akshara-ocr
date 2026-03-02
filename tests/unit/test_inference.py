import os
import sys

# Add root directory to sys.path to resolve 'model' package
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)

import numpy as np
import torch
from model.crnn import CRNN
from model.inference import recognize

def setup_dummy_onnx():
    """
    Creates a dummy ONNX model at the required path for testing inference.py
    """
    model_dir = os.path.join(root_dir, 'model')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    
    # Ensure checkpoints dir exists
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    onnx_path = os.path.join(checkpoints_dir, 'best_model.onnx')
    
    # Simple vocab config to match model initialization
    vocab_size = 96  # Using 95 ASCII chars + 1 blank
    config = {}
    
    # Instantiate untrained model
    model = CRNN(vocab_size=vocab_size, config=config)
    model.eval()
    
    # Dummy input for Export
    dummy_input = torch.randn(1, 1, 32, 128)
    
    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
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
    return onnx_path

def test_recognize_signature_and_execution():
    """
    Tests that the recognize function processes dummy images accurately without crashing.
    """
    # Create 2 dummy images of different widths
    img1 = np.random.rand(32, 128).astype(np.float32)
    img2 = np.random.rand(32, 200).astype(np.float32)
    
    test_inputs = [img1, img2]
    
    onnx_path = None
    try:
        onnx_path = setup_dummy_onnx()
        
        # Execute Function
        results = recognize(test_inputs)
        
        # Verification
        assert isinstance(results, list), f"Expected response to be list, got {type(results)}"
        assert len(results) == len(test_inputs), f"Expected {len(test_inputs)} results, got {len(results)}"
        
        for i, res in enumerate(results):
            assert isinstance(res, str), f"Expected item {i} to be a string, got {type(res)}"
            print(f"Output string {i}: '{res}'")
            
        print("Inference verification completed successfully.")
    finally:
        # Cleanup
        if onnx_path and os.path.exists(onnx_path):
            os.remove(onnx_path)

if __name__ == "__main__":
    test_recognize_signature_and_execution()
