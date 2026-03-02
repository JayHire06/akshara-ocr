import pytest
from unittest.mock import patch, MagicMock
from nlp.postprocessor import correct

@patch('nlp.postprocessor.get_spell_checker')
@patch('nlp.postprocessor.validate')
def test_correct_no_errors(mock_validate, mock_get_spell_checker):
    """Test postprocessor correctly processes text with no spelling/phonetic issues."""
    mock_validate.return_value = True
    
    mock_checker = mock_get_spell_checker.return_value
    mock_checker.correct.side_effect = lambda word, max_edit_distance: word
    
    result = correct("भारत देश", "hi")
    assert result == "भारत देश"
    assert mock_checker.correct.call_count == 2
    assert mock_validate.call_count == 2

@patch('nlp.postprocessor.get_spell_checker')
@patch('nlp.postprocessor.validate')
def test_correct_phonetic_error_fix(mock_validate, mock_get_spell_checker):
    """Test postprocessor uses phonetic correction if word is initially invalid."""
    # Force validate to return False on 'भाारत' and True on 'भारत'
    def validate_side_effect(word, lang):
        return word == "भारत"
    mock_validate.side_effect = validate_side_effect
    
    mock_checker = mock_get_spell_checker.return_value
    mock_checker.correct.side_effect = lambda word, max_edit_distance: word
    
    # phonetic corrector deletes index 1 => 'भारत' which satisfies validate
    result = correct("भाारत", "hi")
    assert result == "भारत"

@patch('nlp.postprocessor.get_spell_checker')
@patch('nlp.postprocessor.validate')
def test_correct_spell_checker_fix(mock_validate, mock_get_spell_checker):
    """Test postprocessor uses spell checker to fix words before phonetic checks."""
    mock_validate.return_value = True
    
    mock_checker = mock_get_spell_checker.return_value
    # mock spell checker translates "भरात" -> "भारत"
    mock_checker.correct.side_effect = lambda word, max_edit_distance: "भारत" if word == "भरात" else word
    
    result = correct("भरात", "hi")
    assert result == "भारत"
    mock_checker.correct.assert_called_with("भरात", max_edit_distance=1)
