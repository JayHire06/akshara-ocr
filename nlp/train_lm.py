import os
import time
from collections import Counter
from pathlib import Path
from nlp.ngram_lm import NgramLM

DATA_DIR = Path(__file__).parent / "data"
DICT_DIR = Path(__file__).parent / "dictionaries"
CORPUS_FILE = DATA_DIR / "hindi_corpus.txt"

def train_and_save_lms():
    print(f"Loading corpus from {CORPUS_FILE}...")
    if not CORPUS_FILE.exists():
        print("Corpus not found! Run python -m nlp.corpus_builder first.")
        return
        
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus_text = f.read()
        
    print(f"Loaded {len(corpus_text)/1024/1024:.2f} MB of text.")
    
    # 1. Build Dictionary
    print("Building dictionary (Top 100,000 words)...")
    words = corpus_text.split()
    word_counts = Counter(words)
    top_words = [word for word, count in word_counts.most_common(100000)]
    
    dict_file = DICT_DIR / "hi_words.txt"
    with open(dict_file, 'w', encoding='utf-8') as f:
        for word in top_words:
            f.write(word + '\n')
    print(f"Saved {len(top_words)} words to {dict_file}")
    
    # Optional pruning to save RAM and disk size
    # We will limit the text size for word trigrams if memory/speed is an issue, but let's train on all
    
    # 2. Train Char N-gram
    print("Training Character Trigram LM...")
    t0 = time.time()
    char_lm = NgramLM(n=3, level='char')
    char_lm.train(corpus_text)
    char_file = DATA_DIR / "char_tri_hi.json"
    char_lm.save(str(char_file))
    print(f"Saved Char LM in {time.time()-t0:.2f}s")
    
    # Ensure memory is freed mostly
    import gc
    del char_lm
    gc.collect()

    # 3. Train Word Bigram
    print("Training Word Bigram LM...")
    t0 = time.time()
    word_bi = NgramLM(n=2, level='word')
    word_bi.train(corpus_text)
    word_bi_file = DATA_DIR / "word_bi_hi.json"
    
    # Prune low count bigrams to save disk space
    pruned_counts = {}
    for ctx, tgts in word_bi.counts.items():
        pruned_tgts = {k: v for k, v in tgts.items() if v > 1}
        if pruned_tgts:
            word_bi.counts[ctx] = pruned_tgts
            
    word_bi.save(str(word_bi_file))
    print(f"Saved Word Bigram LM in {time.time()-t0:.2f}s")
    
    del word_bi
    gc.collect()

if __name__ == "__main__":
    train_and_save_lms()
