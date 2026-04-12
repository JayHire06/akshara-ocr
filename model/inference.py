import numpy as np
import onnxruntime as ort
from typing import List
from model.ctc_decoder import CTCDecoder
import json
import os


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DUMMY_TEST_ONNX_PATH = os.path.join(ROOT_DIR, "test.onnx")


def _get_candidate_model_paths() -> list[str]:
    env_path = os.environ.get("AKSHARA_MODEL_PATH")
    candidates = [
        env_path,
        os.path.join(os.path.dirname(__file__), "checkpoints_200k", "akshara_hindi_v1.onnx"),
        os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.onnx"),
        DUMMY_TEST_ONNX_PATH,
    ]
    return [path for path in candidates if path]


def _load_vocab(class_count: int) -> list[str]:
    vocab_path = os.path.join(ROOT_DIR, "vocab.json")

    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        max_index = max(vocab_dict.values(), default=0)
        if max_index < class_count:
            vocab = [""] * class_count
            for char, idx in vocab_dict.items():
                if idx < class_count:
                    vocab[idx] = char
            return vocab

    # Default ASCII range stub
    fallback = [""] + [chr(i) for i in range(32, 127)]
    if len(fallback) < class_count:
        fallback.extend(["?"] * (class_count - len(fallback)))
    return fallback[:class_count]

def recognize(line_images: List[np.ndarray]) -> List[str]:
    """
    Input: list of float32 arrays, width can be variable. shape of each: (1, 32, W)
    Output: list of raw decoded strings
    """
    session = None
    last_error = None

    for model_path in _get_candidate_model_paths():
        if not os.path.exists(model_path):
            continue

        if model_path == DUMMY_TEST_ONNX_PATH and os.environ.get("AKSHARA_ALLOW_DUMMY_MODEL") != "1":
            # `test.onnx` in this repo is exported by test_export.py from a randomly
            # initialized CRNN(vocab_size=96), so it is not a usable OCR model.
            last_error = RuntimeError(
                "No trained OCR model is available. "
                "The checked-in test.onnx is a dummy export from test_export.py. "
                "Set AKSHARA_MODEL_PATH to a real trained ONNX export."
            )
            continue

        try:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            break
        except Exception as exc:
            last_error = exc

    if session is None:
        raise RuntimeError(str(last_error or "Unable to initialize OCR model"))

    output_shape = session.get_outputs()[0].shape
    class_count = int(output_shape[-1]) if isinstance(output_shape[-1], int) else 96
    decoder = CTCDecoder(vocab=_load_vocab(class_count), blank_id=0)
    
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
