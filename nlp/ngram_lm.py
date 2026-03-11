from collections import defaultdict
import math
from typing import List, Tuple, Dict, Any

class NgramLM:
    def __init__(self, n: int = 3, level: str = 'char', k: float = 0.01):
        """
        Initializes the N-gram Language Model.
        n: The order of the model (e.g., 3 for trigram).
        level: 'char' or 'word'.
        k: Add-k smoothing parameter.
        """
        self.n = n
        self.level = level
        self.k = k
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def train(self, corpus: str):
        """Trains the model on the given corpus string."""
        if self.level == 'char':
            tokens = list(corpus)
        else:
            tokens = corpus.split()
            
        self.vocab.update(tokens)
        
        # Pad tokens
        pad = ['<s>'] * (self.n - 1)
        tokens = pad + tokens + ['</s>']
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tokens[i:i+self.n]
            context = tuple(ngram[:-1])
            target = ngram[-1]
            
            self.counts[context][target] += 1
            self.context_counts[context] += 1

    def score(self, text: str) -> float:
        """Returns the log probability of the text according to the model."""
        if self.level == 'char':
            tokens = list(text)
        else:
            tokens = text.split()
            
        pad = ['<s>'] * (self.n - 1)
        tokens = pad + tokens + ['</s>']
        
        log_prob = 0.0
        vocab_size = len(self.vocab)
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tokens[i:i+self.n]
            context = tuple(ngram[:-1])
            target = ngram[-1]
            
            count = self.counts[context].get(target, 0)
            context_count = self.context_counts[context]
            
            # Add-k smoothing
            prob = (count + self.k) / (context_count + self.k * vocab_size)
            log_prob += math.log(prob)
            
        return log_prob

    def most_likely(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Returns the most likely next tokens given a context."""
        if self.level == 'char':
            context_tokens = tuple(list(context)[-(self.n-1):])
        else:
            context_tokens = tuple(context.split()[-(self.n-1):])
            
        if len(context_tokens) < self.n - 1:
            pad = tuple(['<s>'] * (self.n - 1 - len(context_tokens)))
            context_tokens = pad + context_tokens
            
        next_counts = self.counts.get(context_tokens, {})
        context_count = self.context_counts.get(context_tokens, 0)
        vocab_size = len(self.vocab)
        
        probs = []
        for target, count in next_counts.items():
            prob = (count + self.k) / (context_count + self.k * vocab_size)
            probs.append((target, prob))
            
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]
        
    def save(self, filepath: str):
        import json
        
        # Convert tuple contexts to strings
        counts_serializable = {}
        for ctx, tgts in self.counts.items():
            ctx_str = "|||".join(ctx)
            counts_serializable[ctx_str] = dict(tgts)
            
        context_counts_serializable = {
            "|||".join(ctx): count 
            for ctx, count in self.context_counts.items()
        }
        
        data = {
            "n": self.n,
            "level": self.level,
            "k": self.k,
            "vocab": list(self.vocab),
            "counts": counts_serializable,
            "context_counts": context_counts_serializable
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    def load(self, filepath: str):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.n = data["n"]
        self.level = data["level"]
        self.k = data["k"]
        self.vocab = set(data["vocab"])
        
        self.counts = defaultdict(lambda: defaultdict(int))
        for ctx_str, tgts in data["counts"].items():
            ctx = tuple(ctx_str.split("|||")) if ctx_str else ()
            self.counts[ctx].update(tgts)
            
        self.context_counts = defaultdict(int)
        for ctx_str, count in data["context_counts"].items():
            ctx = tuple(ctx_str.split("|||")) if ctx_str else ()
            self.context_counts[ctx] = count
