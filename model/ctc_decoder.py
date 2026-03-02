# FILE: /model/ctc_decoder.py
import torch
from typing import List

class CTCDecoder:
    def __init__(self, vocab: List[str], blank_id: int = 0):
        self.vocab = vocab
        self.blank_id = blank_id
        
    def convert_to_string(self, tokens: List[int]) -> str:
        res = []
        for t in tokens:
            if t != self.blank_id and t < len(self.vocab):
                res.append(self.vocab[t])
        return "".join(res)
        
    def decode_best_path(self, log_probs: torch.Tensor) -> List[str]:
        # log_probs: (seq_len, batch_size, vocab_size)
        argmax_probs = torch.argmax(log_probs, dim=2) # (seq_len, batch_size)
        argmax_probs = argmax_probs.transpose(0, 1) # (batch_size, seq_len)
        
        decoded_strings = []
        for b in range(argmax_probs.size(0)):
            tokens = argmax_probs[b].tolist()
            prev = self.blank_id
            seq = []
            for t in tokens:
                if t != self.blank_id and t != prev:
                    seq.append(t)
                prev = t
            decoded_strings.append(self.convert_to_string(seq))
            
        return decoded_strings

    def decode_beam_search(self, log_probs: torch.Tensor, beam_width: int = 5) -> List[str]:
        # Basic pure Python beam search
        seq_len, batch_size, vocab_size = log_probs.shape
        probs = torch.exp(log_probs).detach().cpu().numpy()
        
        decoded_strings = []
        for b in range(batch_size):
            beam = [(1.0, [], self.blank_id)]
            for t in range(seq_len):
                new_beam = []
                for score, path, last_token in beam:
                    for v in range(vocab_size):
                        p = probs[t, b, v]
                        new_score = score * p
                        new_path = list(path)
                        new_last_token = v
                        if v != self.blank_id and v != last_token:
                            new_path.append(v)
                        new_beam.append((new_score, new_path, new_last_token))
                
                # Sort by score desc and truncate
                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:beam_width]
                
            best_path = beam[0][1]
            decoded_strings.append(self.convert_to_string(best_path))
            
        return decoded_strings
