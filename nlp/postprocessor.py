import time
from typing import List
from .spell_checker import SpellChecker
from .phonetic_rules import validate
from .yuktakshar import split_conjuncts

_spell_checkers = {}

def get_spell_checker(language: str) -> SpellChecker:
    if language not in _spell_checkers:
        _spell_checkers[language] = SpellChecker(language)
    return _spell_checkers[language]

def attempt_phonetic_correction(word: str, language: str) -> str:
    """Attempts local correction if a word violates phonetic rules."""
    # Try deleting characters one by one
    for i in range(len(word)):
        candidate = word[:i] + word[i+1:]
        if validate(candidate, language):
            return candidate
            
    # Try swapping adjacent characters
    for i in range(len(word) - 1):
        candidate = word[:i] + word[i+1] + word[i] + word[i+2:]
        if validate(candidate, language):
            return candidate
            
    return word # Fallback to original

from .ngram_lm import NgramLM
from pathlib import Path

_lms = {}

def get_lm(language: str) -> NgramLM:
    if language not in _lms:
        lm_path = Path(__file__).parent / "data" / f"char_tri_{language}.json"
        lm = NgramLM(n=3, level='char')
        if lm_path.exists():
            lm.load(str(lm_path))
        _lms[language] = lm
    return _lms[language]

def correct(raw_text: str, language: str) -> str:
    """
    Corrects raw CTC OCR output.
    Step 1: Split into words.
    Step 2: Check spell checker (get all candidates).
    Step 3: Filter by phonetic rules.
    Step 4: Score remaining candidates with LM and pick best.
    """
    checker = get_spell_checker(language)
    lm = get_lm(language)
    words = raw_text.split()
    corrected_words = []
    
    for word in words:
        # Step 2: Spell Check (get all candidates within distance 1)
        candidates = checker.correct(word, max_edit_distance=1, return_all=True)
        if isinstance(candidates, str):
            candidates = [candidates]
            
        # Step 3: Phonetic Check
        valid_candidates = []
        for cand in candidates:
            if validate(cand, language):
                valid_candidates.append(cand)
            else:
                phonetic_corrected = attempt_phonetic_correction(cand, language)
                if phonetic_corrected not in valid_candidates:
                    valid_candidates.append(phonetic_corrected)
                    
        # If no valid candidates after rules, fallback to original
        if not valid_candidates:
            valid_candidates = [word]
            
        # Step 4: Score with LM
        best_cand = valid_candidates[0]
        best_score = float('-inf')
        
        for cand in valid_candidates:
            score = lm.score(cand)
            if score > best_score:
                best_score = score
                best_cand = cand
                
        corrected_words.append(best_cand)
        
    return " ".join(corrected_words)

if __name__ == "__main__":
    # Demo script showing correction on 5 example Hindi OCR errors.
    test_cases = [
        ("\u093Eभारत", "hi"), # Starts with matra -> should delete the matra
        ("हिंन्दी", "hi"),    # Spell check might fix or phonetic will complain about something? Actually invalid sequence?
        ("भाारत", "hi"),    # Double matra might be invalid depending on rule implementation
        ("\u094Dदेश", "hi"),  # Starts with halant
        ("दे\u093Cश", "hi")    # Nukta after vowel sign (invalid)
    ]
    
    # Pre-warm dictionary loading
    get_spell_checker("hi")
    
    print("--- Post-Processor Demo ---")
    
    for error_text, lang in test_cases:
        t0 = time.time()
        corrected = correct(error_text, lang)
        t1 = time.time()
        elapsed_ms = (t1 - t0) * 1000
        print(f"Original: {error_text.encode('unicode_escape').decode()} | Corrected: {corrected.encode('unicode_escape').decode()} | Corrected String: {corrected} | Time: {elapsed_ms:.2f}ms")
