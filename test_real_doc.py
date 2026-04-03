"""
Diagnostic test for preprocess/segment.py against real_doc_03.jpg
Prints: number of lines, words per line, and simulated combined text output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess.segment import segment_page, combine_page_text

IMAGE_PATH = 'data/real_test/hindi/real_doc_03.jpg'

print(f"Running segment_page on: {IMAGE_PATH}\n")
page_words = segment_page(IMAGE_PATH)

print(f"Lines detected: {len(page_words)}")
for i, line_words in enumerate(page_words):
    print(f"  Line {i+1}: {len(line_words)} word(s)")

print()

# Simulate text output using word image sizes as placeholder text
# (a real OCR recognize_fn would return the actual text)
def mock_recognize(word_img):
    w, h = word_img.size
    return f"[w{w}]"

combined_text = combine_page_text(page_words, mock_recognize)

print("Simulated combined output (each [] is a recognized word placeholder):")
print("-" * 60)
print(combined_text)
print("-" * 60)
print(f"\nTotal lines in output: {len(combined_text.splitlines())}")
