# utils/document_detector.py
import re

def detect_document_type(text):
    """
    Genkender dokumenttype baseret på tekstens struktur og indhold.
    
    Args:
        text: Dokumenttekst
        
    Returns:
        Bedste gæt på dokumenttype
    """
    # Trim teksten til de første 5000 tegn for hurtigere analyse
    sample = text[:5000].lower()
    
    # Mønster-matchning for forskellige dokumenttyper
    patterns = {
        "lovtekst": {
            "patterns": [
                r'§\s*\d+', 
                r'stk\.\s*\d+'
            ],
            "subtypes": {
                "ligningsloven": [r'ligningslovens?', r'ligningslov', r'§\s*33\s*a'],
                "personskatteloven": [r'personskattelovens?', r'personskattelov'],
                "kildeskatteloven": [r'kildeskattelovens?', r'kildeskattelov', r'skattepligtig']
            }
        },
        "vejledning": {
            "patterns": [
                r'\d+\.\d+\.\d+\s+[A-Z]',  # Afsnit med nummering som 1.2.3
                r'juridiske?\s+vejledning',
                r'vejledende',
                r'eksempel:'
            ],
            "subtypes": {
                "den_juridiske_vejledning": [r'juridiske?\s+vejledning', r'[A-Z]\.\d+\.\d+'],
                "styresignal": [r'styresignal', r'skatte\s*styrelsens?']
            }
        },
        "cirkulaere": {
            "patterns": [
                r'cirkulære\s+nr\.\s+\d+',
                r'cir\.\s+nr\.\s+\d+',
                r'\d+\.\s+[A-Za-z].*\n\d+\.\d+\.\s+[A-Za-z]'  # Typisk cirkulære-formatering
            ]
        },
        "afgoerelse": {
            "patterns": [
                r'(SKM|TfS|LSR)[.\s]*\d{4}[.\s]*\d+',
                r'kendelse',
                r'afsagt\s+den',
                r'retten\s+i'
            ]
        }
    }
    
    # Point-system til scoring
    scores = {doc_type: 0 for doc_type in patterns.keys()}
    
    # Beregn score for hver dokumenttype
    for doc_type, pattern_data in patterns.items():
        for pattern in pattern_data["patterns"]:
            matches = re.findall(pattern, sample)
            scores[doc_type] += len(matches) * 2  # 2 point per match
        
        # Tjek subtyper hvis relevant
        if "subtypes" in pattern_data:
            for subtype, subpatterns in pattern_data["subtypes"].items():
                for pattern in subpatterns:
                    if re.search(pattern, sample):
                        scores[doc_type] += 5  # 5 ekstra point for subtype match
    
    # Find dokumenttypen med højest score
    best_match = max(scores.items(), key=lambda x: x[1])
    
    # Hvis ingen god match, returner "generisk"
    if best_match[1] == 0:
        return "generisk"
    
    # Find subtype hvis muligt
    doc_type = best_match[0]
    if "subtypes" in patterns[doc_type]:
        subtype_scores = {}
        for subtype, subpatterns in patterns[doc_type]["subtypes"].items():
            subtype_scores[subtype] = 0
            for pattern in subpatterns:
                if re.search(pattern, sample):
                    subtype_scores[subtype] += 5
        
        best_subtype = max(subtype_scores.items(), key=lambda x: x[1])
        if best_subtype[1] > 0:
            return best_subtype[0]
    
    return doc_type