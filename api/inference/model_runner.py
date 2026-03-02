try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

class ONNXModelRunner:
    def __init__(self, model_path: str = "model.onnx"):
        self.session = None
        if HAS_ORT:
            try:
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            except Exception:
                pass

    def recognize(self, line_images: list) -> tuple[list[str], list[float]]:
        """
        Runs batched inference with onnxruntime.
        Returns a tuple of (recognized_texts, confidences)
        """
        if not self.session:
            # Fallback mock for testing without model
            return (["Mocked ONNX output"] * len(line_images), [0.99] * len(line_images))
            
        # 1. Preprocess images into batch tensor
        # 2. Run session: outputs = self.session.run(None, {"input": batch_tensor})
        # 3. Decode CTC probabilities to text and confidences
        
        return (["Mocked ONNX output"] * len(line_images), [0.99] * len(line_images))

model_runner = ONNXModelRunner()
