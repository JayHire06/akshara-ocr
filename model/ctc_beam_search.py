import torch
from collections import defaultdict

def ctc_beam_search_decoder(log_probs_seq, vocab_inv, beam_width=5, blank_id=0):
    """
    Native Python CTC Prefix Beam Search algorithm.
    Extracts the Top-K most probable strings from the neural network dynamically.
    
    Args:
        log_probs_seq: (T, vocab_size) tensor of log probabilities for a SINGLE sequence
        vocab_inv: Dictionary mapping index to character
        beam_width: Number of branches to maintain
    """
    # Beam contains tuples of (prefix_sequence, p_blank, p_non_blank)
    # p_blank: log probability that the prefix ends in a blank
    # p_non_blank: log probability that the prefix ends in a non-blank
    T, V = log_probs_seq.shape
    
    # Initialize with empty prefix
    beam = {(): (0.0, float('-inf'))} 
    
    for t in range(T):
        curr_probs = log_probs_seq[t] # (V,)
        next_beam = defaultdict(lambda: (float('-inf'), float('-inf')))
        
        # We only consider the top characters to avoid massive scaling slowdowns mapping
        top_k_probs, top_k_indices = torch.topk(curr_probs, k=min(V, beam_width * 2))
        
        for prefix, (p_b, p_nb) in beam.items():
            for i in range(len(top_k_indices)):
                char_idx = top_k_indices[i].item()
                char_prob = top_k_probs[i].item()
                
                # If we pick a blank
                if char_idx == blank_id:
                    n_p_b, n_p_nb = next_beam[prefix]
                    
                    # Log-sum-exp trick natively mapping for merging prefixes
                    # Math: log(exp(val1) + exp(val2)) natively built by PyTorch but doing it simply
                    merged_p = torch.logaddexp(torch.tensor(n_p_b), torch.tensor(p_b) + char_prob)
                    merged_p = torch.logaddexp(merged_p, torch.tensor(p_nb) + char_prob)
                    next_beam[prefix] = (merged_p.item(), n_p_nb)
                    continue
                    
                # If we pick a normal character
                char = vocab_inv.get(char_idx, "")
                if not char:
                    continue
                    
                # Extend the prefix
                extended_prefix = prefix + (char,)
                n_p_b, n_p_nb = next_beam[extended_prefix]
                
                # If character is same as the last character and no blank between them
                if len(prefix) > 0 and char == prefix[-1]:
                    # Transition from blank ONLY guarantees identical characters are collapsed into a new independent sequence
                    p_add_from_blank = p_b + char_prob
                    merged_p_nb = torch.logaddexp(torch.tensor(n_p_nb), torch.tensor(p_add_from_blank))
                    next_beam[extended_prefix] = (n_p_b, merged_p_nb.item())
                    
                    # Extending current identically ends the repeating sequence
                    n_p_b_old, n_p_nb_old = next_beam[prefix]
                    merged_p_nb_old = torch.logaddexp(torch.tensor(n_p_nb_old), torch.tensor(p_nb) + char_prob)
                    next_beam[prefix] = (n_p_b_old, merged_p_nb_old.item())
                    
                else: 
                    # Standard transition
                    p_add = torch.logaddexp(torch.tensor(p_b), torch.tensor(p_nb)) + char_prob
                    merged_p_nb = torch.logaddexp(torch.tensor(n_p_nb), p_add)
                    next_beam[extended_prefix] = (n_p_b, merged_p_nb.item())
                    
        # Prune the beam
        beam = dict(sorted(next_beam.items(), key=lambda x: torch.logaddexp(torch.tensor(x[1][0]), torch.tensor(x[1][1])).item(), reverse=True)[:beam_width])
        
    final_candidates = []
    for prefix, (p_b, p_nb) in beam.items():
        total_p = torch.logaddexp(torch.tensor(p_b), torch.tensor(p_nb)).item()
        final_candidates.append(("".join(prefix), total_p))
        
    return final_candidates

def nlp_reranker(raw_candidates, spell_checker, threshold_edit_distance=2):
    """
    The linguistic brain mapping. Evaluates candidate strings across the Native dictionary.
    """
    for candidate_string, prob in raw_candidates:
        if not candidate_string:
            continue
            
        # Parse individual words inside the sentence string mapping
        words = candidate_string.split()
        reranked_sentence = []
        is_perfectly_valid = True
        
        for w in words:
            # Check native dictionary Trie
            correction = spell_checker.correct(w, max_edit_distance=threshold_edit_distance, return_all=False)
            if correction:
               reranked_sentence.append(correction)
            else:
               reranked_sentence.append(w)
               is_perfectly_valid = False
               
        # If we successfully parsed the top candidate against the dictionary, map it out safely natively
        return " ".join(reranked_sentence)
        
    # Fallback to greedy 1st candidate if nothing computes completely
    return "".join(raw_candidates[0][0]) if raw_candidates else ""
