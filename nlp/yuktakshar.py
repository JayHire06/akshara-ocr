def split_conjuncts(word: str) -> list:
    """Splits a Devanagari word into syllables (aksharas) including conjuncts."""
    aksharas = []
    current_akshara = ""
    halant = '\u094D'
    
    for i, char in enumerate(word):
        if char == halant:
            current_akshara += char
            continue
            
        if current_akshara:
            if current_akshara[-1] == halant:
                current_akshara += char
            else:
                aksharas.append(current_akshara)
                current_akshara = char
        else:
            current_akshara = char
            
    if current_akshara:
        aksharas.append(current_akshara)
        
    return aksharas

def reconstruct_conjuncts(aksharas: list) -> str:
    """Reconstructs text from a list of aksharas."""
    return "".join(aksharas)
