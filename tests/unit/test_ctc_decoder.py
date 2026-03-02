import pytest
import torch
from model.ctc_decoder import CTCDecoder

@pytest.fixture
def decoder():
    vocab = ["<blank>", "a", "b", "c", " "]
    return CTCDecoder(vocab=vocab, blank_id=0)

def test_convert_to_string(decoder):
    """Test conversion of token IDs to string, ignoring blanks."""
    tokens = [1, 2, 0, 3, 4, 1]
    result = decoder.convert_to_string(tokens)
    assert result == "abc a"

def test_decode_best_path(decoder):
    """Test greedy decoding (best path) collapses repeats and removes blanks."""
    # seq_len=4, batch_size=1, vocab_size=5
    log_probs = torch.zeros((4, 1, 5))
    log_probs[0, 0, 1] = 10.0  # a
    log_probs[1, 0, 1] = 10.0  # a (repeat, will be collapsed)
    log_probs[2, 0, 0] = 10.0  # blank
    log_probs[3, 0, 2] = 10.0  # b
    
    result = decoder.decode_best_path(log_probs)
    assert result == ["ab"]

def test_decode_beam_search(decoder):
    """Test beam search decoding works and produces expected string."""
    log_probs = torch.zeros((3, 1, 5))
    # Path: a -> blank -> b => "ab"
    log_probs[0, 0, 1] = 10.0
    log_probs[1, 0, 0] = 10.0
    log_probs[2, 0, 2] = 10.0
    
    result = decoder.decode_beam_search(log_probs, beam_width=2)
    assert result == ["ab"]
