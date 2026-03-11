import time
from nlp.postprocessor import correct

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def calculate_crr(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0
    dist = levenshtein(reference, hypothesis)
    cer = dist / len(reference)
    return max(0.0, 1.0 - cer)

def evaluate_postprocessor():
    # 20 mocked examples of CTC errors
    test_data = [
        # (Raw CTC output, Ground Truth)
        ("भारात", "भारत"),                     # Extra 'a' matra
        ("आजाादी", "आजादी"),                    # Double 'a' matra 
        ("हिंन्दी", "हिन्दी"),                    # Extra Anusvara
        ("दे\u093Cश", "देश"),                    # Nukta after vowel
        ("विदय़ालय", "विद्यालय"),                # Wrong conjunct attempt
        ("\u093Eकिताब", "किताब"),                # Stranded Matra at beginning
        ("\u094Dभाषा", "भाषा"),                  # Stranded Halant at beginning
        ("राज्जू", "राजू"),                      # Accidental halant insertion
        ("अंातरिक", "आंतरिक"),                  # Anusvara misplaced
        ("समानताा", "समानता"),                  # Extra matra at end
        ("करतेे", "करते"),                      # Double 'e' matra
        ("संविधानं", "संविधान"),                  # Extra anusvara
        ("प्रका्श", "प्रकाश"),                    # Halant before consonant without following
        ("विज्ञाान", "विज्ञान"),                  # Extra 'a' matra
        ("महत्वपूूर्ण", "महत्वपूर्ण"),              # Extra 'u' matra
        ("आौद्योगिकी", "प्रौद्योगिकी"),              # Swapped base vowel
        ("स्ंाविधान", "संविधान"),                  # Mixed halant + anusvara
        ("सवातंत्र्य", "स्वातन्त्र्य"),                # Missing halants
        ("सुरक्शा", "सुरक्षा"),                    # Phonetic variation error
        ("राषट्रिय", "राष्ट्रीय"),                   # Missing halant in conjunct
    ]

    print("Evaluating NLP Post-Processor Pipeline...\n")
    
    total_crr_before = 0.0
    total_crr_after = 0.0
    total_chars = 0
    total_time_ms = 0.0
    
    print(f"{'Raw CTC Text':<20} | {'Corrected Text':<20} | {'Ground Truth':<20} | {'Change'}")
    print("-" * 80)
    
    for raw, truth in test_data:
        # Measure time
        t0 = time.time()
        corrected = correct(raw, "hi")
        t1 = time.time()
        
        proc_time = (t1 - t0) * 1000
        total_time_ms += proc_time
        
        # Calculate metrics
        crr_before = calculate_crr(truth, raw)
        crr_after = calculate_crr(truth, corrected)
        
        total_crr_before += crr_before * len(truth)
        total_crr_after += crr_after * len(truth)
        total_chars += len(truth)
        
        change_indicator = "✅ Improved" if crr_after > crr_before else ("➖ Same" if crr_after == crr_before else "❌ Worsened")
        
        print(f"{raw:<20} | {corrected:<20} | {truth:<20} | {change_indicator}")

    avg_crr_before = (total_crr_before / total_chars) * 100
    avg_crr_after = (total_crr_after / total_chars) * 100
    avg_proc_time = total_time_ms / len(test_data)
    
    print("\n" + "="*50)
    print("QUANTITATIVE RESULTS:")
    print(f"Total Examples     : {len(test_data)}")
    print(f"CRR Before         : {avg_crr_before:.2f}%")
    print(f"CRR After          : {avg_crr_after:.2f}%")
    print(f"Absolute Improve   : +{(avg_crr_after - avg_crr_before):.2f}%")
    print(f"Post-Proc Speed    : {avg_proc_time:.2f}ms per word")
    print("="*50)

if __name__ == "__main__":
    evaluate_postprocessor()
