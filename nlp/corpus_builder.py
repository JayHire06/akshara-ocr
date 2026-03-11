import os
import urllib.request
import re
import bz2
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HINDI_CORPUS_FILE = DATA_DIR / "hindi_corpus.txt"

# Wikipedia dump url
WIKI_DUMP_URL = "https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"
TARGET_SIZE_MB = 100
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

def clean_wiki_text(text: str) -> str:
    """Removes wiki markup and keeps only Devanagari sentences."""
    if not text:
        return ""
    # Remove templates and tags
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref.*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image|चित्र):.*?\]\]', '', text)
    # Extract text from simple links [[Link|Text]] or [[Link]]
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Keep only Devanagari, basic punctuation and whitespace
    text = re.sub(r'[^\u0900-\u097F\s।,-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_hindi_corpus():
    """Downloads Hindi Wiki dump and parses it until 100MB of clean text is collected."""
    if HINDI_CORPUS_FILE.exists() and HINDI_CORPUS_FILE.stat().st_size >= TARGET_SIZE_BYTES * 0.9:
        print(f"Corpus already exists at {HINDI_CORPUS_FILE} with size {HINDI_CORPUS_FILE.stat().st_size / 1024 / 1024:.2f} MB")
        return

    print(f"Downloading and parsing from {WIKI_DUMP_URL}...")
    request = urllib.request.Request(WIKI_DUMP_URL, headers={'User-Agent': 'Mozilla/5.0'})
    
    bytes_written = 0
    namespace = '{http://www.mediawiki.org/xml/export-0.11/}'
    # Some older or newer dumps might have different namespaces, so we'll just deal with local names
    
    with urllib.request.urlopen(request) as response:
        with bz2.BZ2File(response, 'rb') as uncompressed:
            with open(HINDI_CORPUS_FILE, 'w', encoding='utf-8') as out_file:
                context = ET.iterparse(uncompressed, events=("end",))
                for event, elem in context:
                    if elem.tag.endswith('text'):
                        text = elem.text
                        if text:
                            cleaned = clean_wiki_text(text)
                            if cleaned and len(cleaned) > 50: # Only write decent sized chunks
                                out_file.write(cleaned + "\n")
                                bytes_written += len(cleaned.encode('utf-8')) + 1
                                
                                if bytes_written >= TARGET_SIZE_BYTES:
                                    print(f"Reached target size: {bytes_written / 1024 / 1024:.2f} MB")
                                    break
                    elem.clear() # Free memory
                    
    print(f"Finished writing corpus: {HINDI_CORPUS_FILE}")

if __name__ == "__main__":
    build_hindi_corpus()
