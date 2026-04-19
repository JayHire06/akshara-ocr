# FILE: /model/ctc_decoder.py
import math
from typing import List

import torch


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


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

    def decode_prefix_beam(self, log_probs: torch.Tensor, beam_width: int = 10) -> List[str]:
        """Proper CTC prefix beam search (Graves 2006).

        Maintains, per prefix s, two probabilities: P_b(s) (ending in blank)
        and P_nb(s) (ending in non-blank). Merges equivalent prefixes via
        log-sum-exp. Works in log space to avoid underflow on long sequences.

        Stronger than `decode_beam_search` above, which does not track the
        blank/non-blank split and does not merge paths that collapse to the
        same text — both of which are required for CTC beam to actually
        improve over greedy.

        log_probs: (seq_len, batch_size, vocab_size) log-probabilities.
        Returns: list of decoded strings, one per batch element.
        """
        seq_len, batch_size, vocab_size = log_probs.shape
        lp_np = log_probs.detach().cpu().numpy()
        NEG = -math.inf
        results: List[str] = []

        for b in range(batch_size):
            # prefix (tuple[int, ...]) -> (log_Pb, log_Pnb)
            beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG)}

            for t in range(seq_len):
                new_beams: dict[tuple, tuple[float, float]] = {}

                def _add(key: tuple, pb_add: float, pnb_add: float) -> None:
                    cb, cnb = new_beams.get(key, (NEG, NEG))
                    new_beams[key] = (_logsumexp(cb, pb_add), _logsumexp(cnb, pnb_add))

                for prefix, (pb, pnb) in beams.items():
                    total = _logsumexp(pb, pnb)
                    lp_blank = lp_np[t, b, self.blank_id]
                    _add(prefix, total + lp_blank, NEG)

                    last = prefix[-1] if prefix else -1
                    for c in range(vocab_size):
                        if c == self.blank_id:
                            continue
                        lp = lp_np[t, b, c]
                        if c == last:
                            # Same char as last: extending via blank creates a
                            # new repeat token; staying on same prefix is the
                            # CTC-collapse case.
                            _add(prefix, NEG, pnb + lp)
                            _add(prefix + (c,), NEG, pb + lp)
                        else:
                            _add(prefix + (c,), NEG, total + lp)

                # Prune to beam_width by total prefix probability
                scored = [
                    (key, val, _logsumexp(val[0], val[1]))
                    for key, val in new_beams.items()
                ]
                scored.sort(key=lambda x: x[2], reverse=True)
                beams = {k: v for k, v, _ in scored[:beam_width]}

            best_prefix, _ = max(
                beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1])
            )
            results.append(self.convert_to_string(list(best_prefix)))

        return results
