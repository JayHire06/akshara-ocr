import os
import urllib.request
import re
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HINDI_CORPUS_FILE = DATA_DIR / "hindi_corpus.txt"

WIKI_DUMP_URL = "https://raw.githubusercontent.com/AI4Bharat/indic-nlp-corpus/master/hindi/sample.txt"

def clean_text(text: str) -> str:
    """Preprocesses text by removing non-Devanagari characters (except spaces)."""
    # Keep only Devanagari characters and whitespace
    cleaned = re.sub(r'[^\u0900-\u097F\s]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def build_hindi_corpus():
    """Downloads a sample Hindi corpus if it doesn't exist and cleans it."""
    if HINDI_CORPUS_FILE.exists():
        print(f"Corpus already exists at {HINDI_CORPUS_FILE}")
        return

    print("Downloading corpus...")
    try:
        response = urllib.request.urlopen(WIKI_DUMP_URL, timeout=10)
        content = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download from {WIKI_DUMP_URL}: {e}")
        print("Falling back to small included sample...")
        content = "भारत एक महान देश है। हिंदी हमारी भाषा है।"

    cleaned_content = clean_text(content)
    
    with open(HINDI_CORPUS_FILE, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    print(f"Corpus saved to {HINDI_CORPUS_FILE}")

if __name__ == "__main__":
    build_hindi_corpus()
