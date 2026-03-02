from typing import List

# Unicode ranges
RANGES = {
    "hi": (0x0900, 0x097F), # Devanagari
    "ta": (0x0B80, 0x0BFF), # Tamil
    "bn": (0x0980, 0x09FF), # Bengali
}

def get_vocab(lang_code: str) -> List[str]:
    """Generates a list of valid characters for a given language code based on its Unicode block."""
    if lang_code not in RANGES:
        raise ValueError(f"Language {lang_code} not supported or no range defined.")
    start, end = RANGES[lang_code]
    return [chr(code) for code in range(start, end + 1)]
