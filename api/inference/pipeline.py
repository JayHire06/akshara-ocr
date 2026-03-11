import sys
sys.path.insert(0, r'c:\Users\jayhi\Downloads\akshara-ocr')
from model.inference_page import ocr_page

def run_ocr_pipeline(image_path: str) -> dict:
    try:
        text = ocr_page(image_path)
        word_count = len(text.split())
        return {
            "status": "done",
            "text": text,
            "confidence": 0.70,
            "word_count": word_count,
            "processing_time_ms": 0
        }
    except Exception as e:
        return {
            "status": "error",
            "text": str(e),
            "confidence": 0.0,
            "word_count": 0,
            "processing_time_ms": 0
        }
