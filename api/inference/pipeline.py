import numpy as np
from PIL import Image

from model.inference import recognize
from preprocess.segment import segment_page


def _prepare_word_tensor(word_img: Image.Image) -> np.ndarray:
    grayscale = word_img.convert("L")
    arr = np.array(grayscale, dtype=np.float32) / 255.0
    return arr


def run_ocr_pipeline(image_path: str) -> dict:
    try:
        page_words = segment_page(image_path)
        recognized_lines: list[str] = []

        for line_words in page_words:
            if not line_words:
                continue

            prepared_words = [_prepare_word_tensor(word_img) for word_img in line_words]
            predictions = recognize(prepared_words)
            line_text = " ".join(word for word in predictions if word.strip())

            if line_text:
                recognized_lines.append(line_text)

        text = "\n".join(recognized_lines).strip()

        return {
            "status": "done",
            "text": text,
            "confidence": 0.70 if text else 0.0,
            "word_count": len(text.split()) if text else 0,
            "processing_time_ms": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "text": str(exc),
            "confidence": 0.0,
            "word_count": 0,
            "processing_time_ms": 0,
        }
