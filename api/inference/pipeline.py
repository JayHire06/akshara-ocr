# API Inference Pipeline
# Orchestrates: preprocess -> model -> postprocess -> return result

from typing import Tuple

from preprocess.pipeline import preprocess
from model.inference import recognize
from nlp.postprocessor import correct

def run_inference_pipeline(image, language: str) -> Tuple[str, float, int]:
    """
    Orchestrates the complete OCR operations in sequence.
    Normally called by Celery task.
    """
    # 3. Call preprocess(image, language)
    line_images = preprocess(image, language)

    # 4. Call recognize(line_images)
    res = recognize(line_images)
    if isinstance(res, tuple) and len(res) == 2:
        lines, confidences = res
    else:
        lines = res
        confidences = [0.99] * len(lines)

    # 5. Join recognized lines with newline
    raw_text = "\n".join(lines)
    
    # 6. Call correct(raw_text, language)
    final_text = correct(raw_text, language)

    # 7. Compute confidence
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    word_count = len(final_text.split())

    return final_text, mean_confidence, word_count
