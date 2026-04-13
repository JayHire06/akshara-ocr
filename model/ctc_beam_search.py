import torch
import math
from collections import defaultdict

def log_sum_exp(a, b):
    """Semi-stable logsumexp in native python to avoid tensor creation overhead."""
    if a == float("-inf"): return b
    if b == float("-inf"): return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def greedy_decoder(log_probs_seq, vocab_inv, blank_id=0):
    """
    Lightning fast greedy decoder for bulk validation.
    Args:
        log_probs_seq: (T, V) tensor or array of log probabilities
        vocab_inv: Dictionary mapping index to character
    """
    arg_maxes = torch.argmax(log_probs_seq, dim=-1)
    decode = []
    for i in range(len(arg_maxes)):
        idx = arg_maxes[i].item()
        if idx != blank_id:
            if i > 0 and idx == arg_maxes[i-1].item():
                continue
            char = vocab_inv.get(idx, "")
            if char:
                decode.append(char)
    return "".join(decode)

def ctc_beam_search_decoder(log_probs_seq, vocab_inv, beam_width=5, blank_id=0):
    """
    Optimized Python CTC Prefix Beam Search algorithm.
    Refactored to minimize tensor creation overhead and use fast math.
    """
    T, V = log_probs_seq.shape
    
    # Initialize with empty prefix: (p_blank, p_non_blank)
    beam = {(): (0.0, float('-inf'))} 
    
    # Pre-calculate top-k indices and values to avoid redundant torch operations
    top_k_vals, top_k_idxs = torch.topk(log_probs_seq, k=min(V, beam_width * 2), dim=-1)
    top_k_vals = top_k_vals.tolist()
    top_k_idxs = top_k_idxs.tolist()
    
    for t in range(T):
        next_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
        
        # Current time step top k
        t_indices = top_k_idxs[t]
        t_probs = top_k_vals[t]
        
        for prefix, (p_b, p_nb) in beam.items():
            for i in range(len(t_indices)):
                char_idx = t_indices[i]
                char_prob = t_probs[i]
                
                # If we pick a blank
                if char_idx == blank_id:
                    n_p_b, n_p_nb = next_beam[prefix]
                    new_p_b = log_sum_exp(n_p_b, log_sum_exp(p_b, p_nb) + char_prob)
                    next_beam[prefix] = (new_p_b, n_p_nb)
                    continue
                    
                char = vocab_inv.get(char_idx, "")
                if not char: continue
                    
                extended_prefix = prefix + (char,)
                n_p_b, n_p_nb = next_beam[extended_prefix]
                
                if len(prefix) > 0 and char == prefix[-1]:
                    # Case 1: Identical char with blank between them (new prefix)
                    new_p_nb = log_sum_exp(n_p_nb, p_b + char_prob)
                    next_beam[extended_prefix] = (n_p_b, new_p_nb)
                    
                    # Case 2: Identical char without blank (collapse into existing prefix)
                    n_p_b_ex, n_p_nb_ex = next_beam[prefix]
                    new_p_nb_ex = log_sum_exp(n_p_nb_ex, p_nb + char_prob)
                    next_beam[prefix] = (n_p_b_ex, new_p_nb_ex)
                else: 
                    # Standard transition
                    new_p_nb = log_sum_exp(n_p_nb, log_sum_exp(p_b, p_nb) + char_prob)
                    next_beam[extended_prefix] = (n_p_b, new_p_nb)
                    
        # Prune the beam
        beam = dict(sorted(next_beam.items(), key=lambda x: log_sum_exp(x[1][0], x[1][1]), reverse=True)[:beam_width])
        
    final_candidates = []
    for prefix, (p_b, p_nb) in beam.items():
        total_p = log_sum_exp(p_b, p_nb)
        final_candidates.append(("".join(prefix), total_p))
        
    return final_candidates

def nlp_reranker(raw_candidates, spell_checker, threshold_edit_distance=2):
    """
    Reranks Beam Search candidates using the linguistic spell checker Trie.
    """
    if not raw_candidates: return ""
    
    # We only rerank the top few choices to keep latency low
    for candidate_string, _ in raw_candidates[:max(1, len(raw_candidates)//2)]:
        if not candidate_string: continue
        
        words = candidate_string.split()
        reranked_words = []
        all_found = True
        
        for w in words:
            correction = spell_checker.correct(w, max_edit_distance=threshold_edit_distance, return_all=False)
            if correction:
               reranked_words.append(correction)
            else:
               reranked_words.append(w)
               all_found = False
               
        if all_found:
            return " ".join(reranked_words)
            
    # Fallback to the most probable raw candidate
    return raw_candidates[0][0]
