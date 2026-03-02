def is_valid_devanagari(text: str) -> bool:
    """Validates Devanagari phonotactic rules."""
    matras = set(chr(c) for c in range(0x093E, 0x094D))
    halant = '\u094D'
    nukta = '\u093C'
    anusvara = '\u0902'
    chandrabindu = '\u0901'
    visarga = '\u0903'
    
    consonants = set(chr(c) for c in range(0x0915, 0x093A))
    consonants.update([chr(c) for c in range(0x0958, 0x0960)])
    
    vowels = set(chr(c) for c in range(0x0904, 0x0915))
    
    for i, char in enumerate(text):
        prev = text[i-1] if i > 0 else None
        
        # Rule 1: Matra constraint
        if char in matras:
            if i == 0 or prev in matras:
                return False
            # Matra must follow a consonant
            if prev not in consonants and prev != nukta: 
                return False
                
        # Rule 2: Halant rule
        if char == halant:
            if i == 0 or prev in matras or prev in vowels:
                return False
            if prev not in consonants and prev != nukta:
                return False
                
        # Rule 3: Nukta rule
        if char == nukta:
            if i == 0 or prev in matras or prev in vowels:
                return False
            if prev not in consonants:
                return False
                
        # Rule 4 & 5: Anusvara/Chandrabindu/Visarga
        if char in [anusvara, chandrabindu, visarga]:
            if i == 0:
                return False
            # Can only appear after a vowel or matra (or consonant with inherent a)
            # Technically after consonant is assumed to have inherent a
            if prev not in vowels and prev not in matras and prev not in consonants:
                return False
                
    return True

def is_valid_tamil(text: str) -> bool:
    """Validates Tamil phonotactic rules."""
    vowel_signs = set(chr(c) for c in range(0x0BBE, 0x0BCD))
    virama = '\u0BCD'
    consonants = set(chr(c) for c in range(0x0B95, 0x0BBA))
    vowels = set(chr(c) for c in range(0x0B85, 0x0B95))
    
    for i, char in enumerate(text):
        prev = text[i-1] if i > 0 else None
        
        if char in vowel_signs:
            if i == 0 or prev in vowel_signs:
                return False
            if prev not in consonants:
                return False
                
        if char == virama:
            if i == 0 or prev in vowels or prev in vowel_signs:
                return False
            if prev not in consonants:
                return False
                
    return True

def is_valid_bengali(text: str) -> bool:
    """Validates Bengali phonotactic rules."""
    vowel_signs = set(chr(c) for c in range(0x09BE, 0x09CD))
    virama = '\u09CD'
    nukta = '\u09BC'
    consonants = set(chr(c) for c in range(0x0995, 0x09BA))
    vowels = set(chr(c) for c in range(0x0985, 0x0995))
    
    for i, char in enumerate(text):
        prev = text[i-1] if i > 0 else None
        
        if char in vowel_signs:
            if i == 0 or prev in vowel_signs:
                return False
            if prev not in consonants and prev != nukta:
                return False
                
        if char == virama:
            if i == 0 or prev in vowels or prev in vowel_signs:
                return False
            if prev not in consonants and prev != nukta:
                return False
                
        if char == nukta:
            if i == 0 or prev in vowels or prev in vowel_signs:
                return False
            if prev not in consonants:
                return False
                
    return True

def validate(text: str, language: str) -> bool:
    if language == "hi":
        return is_valid_devanagari(text)
    elif language == "ta":
        return is_valid_tamil(text)
    elif language == "bn":
        return is_valid_bengali(text)
    return True
