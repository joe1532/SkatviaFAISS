import re
import streamlit as st
import uuid
import json

def validate_chunks(chunks, context_summary):
    """
    Validerer indekserede chunks i forhold til kontekstopsummering med forbedret juridisk validering.
    
    Args:
        chunks: Liste af chunks
        context_summary: Kontekstopsummering
    
    Returns:
        Valideringsresultater
    """
    validation_results = {
        "missing_notes": [],
        "missing_paragraphs": [],
        "inconsistencies": [],
        "missing_legal_exceptions": [],
        "missing_person_groups": [],
        "context_issues": [],
        "overall_status": "success",
        "overall_score": 10.0  # Start med perfekt score og træk fra for problemer
    }
    
    # 1. Tjek om alle noter fra context_summary findes i chunks
    if "summary" in context_summary and "notes_overview" in context_summary["summary"]:
        expected_notes = set(context_summary["summary"]["notes_overview"].keys())
        found_notes = set(c["metadata"].get("note_number", "") for c in chunks if c["metadata"].get("is_note", False))
        
        missing_notes = expected_notes - found_notes
        if missing_notes:
            validation_results["missing_notes"] = list(missing_notes)
            validation_results["overall_status"] = "warning"
            validation_results["overall_score"] -= len(missing_notes) * 0.2  # 0.2 point pr. manglende note
            
    # 2. Tjek om alle paragraffer/stykker fra context_summary findes i chunks
    if "summary" in context_summary and "document_structure" in context_summary["summary"]:
        # Normalisér formatering for sammenligninger
        def normalize_format(text):
            if not text:
                return ""
            # Normalisér mellemrum og store/små bogstaver
            text = re.sub(r'\s+', ' ', text.strip().lower())
            # Normalisér paragraf og stykke formatering
            text = re.sub(r'§\s*(\d+[a-z]?)', r'§ \1', text)
            text = re.sub(r'stk\.?\s*(\d+)', r'stk. \1', text)
            return text
            
        # Udpak forventet struktur med formatering fra context_summary
        expected_structure = {}
        for para, stykker in context_summary["summary"]["document_structure"].items():
            norm_para = normalize_format(para)
            expected_structure[norm_para] = True
            
            # For hver paragraf, tjek om der er stykker
            if isinstance(stykker, list):
                for stykke in stykker:
                    expected_structure[f"{norm_para}, {normalize_format(stykke)}"] = True
        
        # Udpak faktisk struktur fra chunks
        found_structure = {}
        for chunk in chunks:
            if not chunk["metadata"].get("is_note", False):
                para = normalize_format(chunk["metadata"].get("paragraph", ""))
                stykke = normalize_format(chunk["metadata"].get("stykke", ""))
                
                if para:
                    found_structure[para] = True
                    if stykke:
                        found_structure[f"{para}, {stykke}"] = True
        
        # Find manglende struktur
        missing_paragraphs = []
        for structure in expected_structure:
            if structure not in found_structure:
                missing_paragraphs.append(structure)
        
        if missing_paragraphs:
            validation_results["missing_paragraphs"] = missing_paragraphs
            validation_results["overall_status"] = "warning"
            validation_results["overall_score"] -= len(missing_paragraphs) * 0.1  # 0.1 point pr. manglende paragraf
    
    # 3. Tjek for juridiske undtagelser nævnt i context_summary
    if "summary" in context_summary and "legal_exceptions" in context_summary["summary"]:
        expected_exceptions = []
        
        # Udpak forventede undtagelser
        for exception_entry in context_summary["summary"]["legal_exceptions"]:
            if isinstance(exception_entry, dict):
                expected_exceptions.append(exception_entry.get("exception", "").lower())
            elif isinstance(exception_entry, str):
                expected_exceptions.append(exception_entry.lower())
        
        # Tjek om undtagelserne er fundet i chunks
        found_exceptions = set()
        for chunk in chunks:
            for exception in chunk["metadata"].get("legal_exceptions", []):
                if isinstance(exception, str):
                    found_exceptions.add(exception.lower())
        
        # Find manglende undtagelser
        missing_exceptions = []
        for exception in expected_exceptions:
            # Tjek om en variant af undtagelsen er fundet
            if not any(exception in found_exc or found_exc in exception for found_exc in found_exceptions):
                missing_exceptions.append(exception)
        
        if missing_exceptions:
            validation_results["missing_legal_exceptions"] = missing_exceptions
            validation_results["overall_status"] = "warning"
            validation_results["overall_score"] -= len(missing_exceptions) * 0.15  # 0.15 point pr. manglende undtagelse
    
    # 4. Tjek for manglende persongrupper
    if "summary" in context_summary and "target_groups" in context_summary["summary"]:
        expected_groups = [group.lower() for group in context_summary["summary"]["target_groups"]]
        
        # Tjek om persongrupperne er fundet i chunks
        found_groups = set()
        for chunk in chunks:
            for group in chunk["metadata"].get("affected_groups", []):
                found_groups.add(group.lower())
        
        # Find manglende persongrupper
        missing_groups = []
        for group in expected_groups:
            # Tjek om en variant af gruppen er fundet
            if not any(group in found_grp or found_grp in group for found_grp in found_groups):
                missing_groups.append(group)
        
        if missing_groups:
            validation_results["missing_person_groups"] = missing_groups
            validation_results["overall_status"] = "warning"
            validation_results["overall_score"] -= len(missing_groups) * 0.1  # 0.1 point pr. manglende persongruppe
    
    # 5. Tjek for kontekstproblemer
    context_issues = find_context_issues(chunks)
    if context_issues:
        validation_results["context_issues"] = context_issues
        validation_results["overall_status"] = "warning"
        validation_results["overall_score"] -= len(context_issues) * 0.25  # 0.25 point pr. kontekstproblem
    
    # 6. Tjek chunk-størrelser
    size_stats = analyze_chunk_sizes(chunks)
    validation_results["size_stats"] = size_stats
    
    # Træk fra for ekstremt små eller store chunks
    pct_extreme_sizes = size_stats["pct_too_small"] + size_stats["pct_too_large"]
    if pct_extreme_sizes > 10:  # Mere end 10% af chunks har problematisk størrelse
        validation_results["overall_score"] -= (pct_extreme_sizes - 10) * 0.02  # 0.02 point pr. procentpoint over 10%
    
    # 7. Tjek for redundans i chunks
    redundancy_score = check_redundancy(chunks)
    validation_results["redundancy_score"] = redundancy_score
    
    if redundancy_score > 0.2:  # Mere end 20% redundans
        validation_results["overall_score"] -= (redundancy_score - 0.2) * 10  # Træk fra baseret på redundans
    
    # Afrund score og sæt nedre grænse til 0
    validation_results["overall_score"] = max(0, round(validation_results["overall_score"], 1))
    
    return validation_results

def find_context_issues(chunks):
    """
    Identificerer kontekstproblemer i chunks, fx hvor en chunk refererer til noget uden kontekst.
    """
    issues = []
    
    # Problematiske udtryk der kan indikere manglende kontekst
    context_patterns = [
        r'\b(som nævnt|ovenfor nævnte|denne|disse)\b',
        r'\b(det|dette|sådan[nt]?)\s+(?!er|har|kan|vil|må)',
        r'\b(den|de)\s+(?:omtalt|nævnt)',
        r'\b(derfor|således|herefter|følgelig)\b'
    ]
    
    for i, chunk in enumerate(chunks):
        content = chunk["content"].lower()
        metadata = chunk["metadata"]
        
        for pattern in context_patterns:
            if re.search(pattern, content):
                # Tjek om der er andre chunks med samme section_id før denne
                has_preceding_context = False
                
                # Find chunks med samme section og subsection der kommer før i dokumentet
                for other_chunk in chunks[:i]:  # Chunks før denne
                    if (other_chunk["metadata"].get("section") == metadata.get("section") and
                        other_chunk["metadata"].get("subsection") == metadata.get("subsection")):
                        has_preceding_context = True
                        break
                
                if not has_preceding_context:
                    match = re.search(pattern, content)
                    context_reference = match.group(0)
                    
                    issues.append({
                        "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                        "issue": f"Reference uden kontekst: '{context_reference}'",
                        "section": metadata.get("section", ""),
                        "subsection": metadata.get("subsection", ""),
                        "severity": "medium"
                    })
                    break  # Kun rapportér én kontekstfejl pr. chunk
    
    return issues

def analyze_chunk_sizes(chunks):
    """
    Analyserer fordelingen af chunk-størrelser.
    """
    sizes = [len(chunk["content"]) for chunk in chunks]
    
    if not sizes:
        return {
            "min_size": 0,
            "max_size": 0,
            "avg_size": 0,
            "median_size": 0,
            "pct_too_small": 0,
            "pct_too_large": 0
        }
    
    # Definer grænser for "for lille" og "for stor"
    too_small = 200  # Under 200 tegn
    too_large = 2000  # Over 2000 tegn
    
    # Beregn statistik
    min_size = min(sizes)
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    
    # Sorter for median
    sorted_sizes = sorted(sizes)
    middle = len(sorted_sizes) // 2
    median_size = sorted_sizes[middle] if len(sorted_sizes) % 2 == 1 else (sorted_sizes[middle-1] + sorted_sizes[middle]) / 2
    
    # Beregn procentdele
    pct_too_small = sum(1 for s in sizes if s < too_small) / len(sizes) * 100
    pct_too_large = sum(1 for s in sizes if s > too_large) / len(sizes) * 100
    
    return {
        "min_size": min_size,
        "max_size": max_size,
        "avg_size": round(avg_size, 1),
        "median_size": median_size,
        "pct_too_small": round(pct_too_small, 1),
        "pct_too_large": round(pct_too_large, 1)
    }

def check_redundancy(chunks):
    """
    Beregner en redundansscore (0-1) baseret på gentagelser i chunks.
    """
    if len(chunks) <= 1:
        return 0.0
    
    # Ekstrahér indhold og chunks_ids
    chunk_contents = []
    for chunk in chunks:
        # Tag kun de første 200 tegn for effektivitet
        chunk_contents.append(chunk["content"][:200].lower())
    
    # Beregn lighed mellem alle par af chunks
    similarity_count = 0
    comparisons = 0
    
    for i in range(len(chunk_contents)):
        for j in range(i+1, len(chunk_contents)):
            # Simpel lighed baseret på fælles substrings
            content_i = chunk_contents[i]
            content_j = chunk_contents[j]
            
            # Hvis en er indeholdt i den anden, høj lighed
            if content_i in content_j or content_j in content_i:
                similarity_count += 1
            # Ellers tjek for delvise overlap
            else:
                # Del i 3-grams og tjek overlap
                n = 3
                grams_i = set(content_i[k:k+n] for k in range(len(content_i)-n+1))
                grams_j = set(content_j[k:k+n] for k in range(len(content_j)-n+1))
                
                if grams_i and grams_j:  # Undgå division med nul
                    overlap = len(grams_i.intersection(grams_j)) / min(len(grams_i), len(grams_j))
                    
                    if overlap > 0.5:  # Over 50% overlap betragtes som redundans
                        similarity_count += overlap
            
            comparisons += 1
    
    # Beregn gennemsnitlig redundansscore
    if comparisons == 0:
        return 0.0
    
    return similarity_count / comparisons

def repair_missing_paragraphs(chunks, context_summary, validation_results, preserved_content=None):
    """
    Reparerer manglende paragraffer og stykker med forbedret logik.
    """
    # Håndter tilfælde hvor validation_results er None
    if validation_results is None:
        print("Advarsel: validation_results er None i repair_missing_paragraphs")
        return chunks.copy()  # Returnér bare en kopi af chunks uden ændringer
    
    updated_chunks = chunks.copy()
    
    if "missing_paragraphs" not in validation_results or not validation_results["missing_paragraphs"]:
        return updated_chunks
        
    # Hent paragraf/stykke struktur fra kontekst
    expected_structure = {}
    if "summary" in context_summary and "document_structure" in context_summary["summary"]:
        for para, stykker in context_summary["summary"]["document_structure"].items():
            if isinstance(stykker, list):
                expected_structure[para] = stykker
    
    # Identificer hvilke paragraffer og stykker der mangler
    missing_paras = set()
    missing_stykker = {}
    
    for missing in validation_results["missing_paragraphs"]:
        if missing is None:
            continue
        if "," in missing:  # Det er paragraf + stykke
            parts = missing.split(",", 1)
            para = parts[0].strip()
            stykke = parts[1].strip()
            missing_paras.add(para)
            if para not in missing_stykker:
                missing_stykker[para] = []
            missing_stykker[para].append(stykke)
        else:  # Det er kun paragraf
            missing_paras.add(missing)
    
    # For hvert manglende stykke, forsøg at finde det i originalteksten
    if preserved_content and "paragraphs" in preserved_content:
        for para, stykker in missing_stykker.items():
            # Find paragraffen i det bevarede indhold
            para_content = None
            for p_key, p_content in preserved_content["paragraphs"].items():
                normalized_key = re.sub(r'\s+', ' ', p_key.lower())
                normalized_para = re.sub(r'\s+', ' ', para.lower())
                
                if normalized_para in normalized_key:
                    para_content = p_content
                    break
            
            if para_content:
                for stykke in stykker:
                    # Prøv at finde stykket i paragrafteksten
                    stykke_pattern = re.compile(rf'{re.escape(stykke)}(.*?)(?=Stk\.|$)', re.DOTALL | re.IGNORECASE)
                    matches = stykke_pattern.findall(para_content)
                    
                    if matches:
                        # Skab et nyt chunk for det manglende stykke
                        content = f"{para} {stykke} {matches[0].strip()}"
                        chunk = {
                            "content": content,
                            "metadata": {
                                "doc_id": chunks[0]["metadata"]["doc_id"] if chunks else f"dok_{uuid.uuid4().hex[:8]}",
                                "doc_type": "juridisk_vejledning",
                                "paragraph": para,
                                "stykke": stykke,
                                "concepts": [],
                                "law_references": [],
                                "case_references": [],
                                "affected_groups": [],
                                "legal_exceptions": [],
                                "theme": "Rekonstrueret",
                                "subtheme": "Manglende stykke",
                                "is_note": False,
                                "chunk_type": "text",
                                "chunk_id": f"rekonstrueret_{para}_{stykke}",
                                "chunk_position": "rekonstrueret",
                                "reconstructed": True,
                                "complexity": "moderat",
                                "legal_status": "gældende"
                            }
                        }
                        updated_chunks.append(chunk)
    
    # Hvis vi ikke kunne rekonstruere, skab simple placeholders for manglende struktur
    for missing in validation_results["missing_paragraphs"]:
        if missing is None:
            continue
            
        # Tjek om vi allerede har rekonstrueret dette eller om det findes
        existing = False
        for chunk in updated_chunks:
            para = chunk["metadata"].get("paragraph", "")
            stykke = chunk["metadata"].get("stykke", "")
            
            if para is None:
                para = ""
            if stykke is None:
                stykke = ""
                
            chunk_key = para
            if stykke:
                chunk_key = f"{para}, {stykke}"
                
            if chunk_key.lower() == missing.lower():
                existing = True
                break
        
        if not existing:
            # Skab et placeholder-chunk
            if "," in missing:
                parts = missing.split(",", 1)
                para = parts[0].strip()
                stykke = parts[1].strip()
            else:
                para = missing
                stykke = None
                
            # Forsøg at tilføje relevant indhold fra kontekstopsummering
            placeholder_content = f"Placeholder for {missing}"
            
            if "summary" in context_summary:
                # Søg efter relevant beskrivelse i summary
                structure = context_summary["summary"].get("document_structure", {})
                if para in structure and isinstance(structure[para], dict):
                    placeholder_content = structure[para].get("description", placeholder_content)
            
            placeholder_chunk = {
                "content": placeholder_content,
                "metadata": {
                    "doc_id": chunks[0]["metadata"]["doc_id"] if chunks else f"dok_{uuid.uuid4().hex[:8]}",
                    "doc_type": "juridisk_vejledning",
                    "paragraph": para,
                    "stykke": stykke,
                    "concepts": [],
                    "law_references": [],
                    "case_references": [],
                    "affected_groups": [],
                    "legal_exceptions": [],
                    "theme": "Rekonstrueret",
                    "subtheme": "Manglende paragraf/stykke",
                    "is_note": False,
                    "chunk_type": "placeholder",
                    "chunk_id": f"placeholder_{para}_{stykke if stykke else ''}",
                    "chunk_position": "placeholder",
                    "reconstructed": True,
                    "complexity": "moderat",
                    "legal_status": "gældende"
                }
            }
            updated_chunks.append(placeholder_chunk)
    
    return updated_chunks

def extract_legal_exceptions_from_content(chunks):
    """Udtrækker juridiske undtagelser og specialregler fra chunks med forbedret mønstergenkendelse"""
    updated_chunks = []
    
    # Mønstre der kan indikere undtagelser og specialregler
    exception_patterns = [
        r'(?:undtagelse|særregel|specialregel)[^\.;,]*?(?=\.|;|$)',
        r'(?:gælder ikke|finder ikke anvendelse)[^\.;,]*?(?=\.|;|$)',
        r'(?:medmindre|dog ikke|undtaget herfra er)[^\.;,]*?(?=\.|;|$)',
        r'(?:uanset|til trods for)[^\.;,]*?(?=\.|;|$)',
        r'(?:Hovedreglen|Udgangspunktet).*?(?:men|dog)[^\.;,]*?(?=\.|;|$)'
    ]
    
    # Specifikke undtagelser i skatteret
    specific_exceptions = {
        r'\bgrænse(?:gænger|pendler)': ["Grænsegængerreglerne", "grænsegænger"],
        r'§§?\s*5\s*[A-D]': ["KSL §§ 5 A-D"],
        r'\b42[\s-]*dages?\b': ["42-dages reglen"],
        r'\bseks\s+m[åa]neder\b|\b6\s+m[åa]neder\b': ["6-måneders reglen"]
    }
    
    # Mønstrene for persongrupper der kan være omfattet af særregler
    person_groups = {
        "grænsegænger": ["grænsegænger", "pendler over grænsen"],
        "udsendt medarbejder": ["udsendt medarbejder", "udstationeret"],
        "søfolk": ["søfolk", "søfarende", "skibspersonale"],
        "selvstændige": ["selvstændige", "selvstændig erhvervsdrivende"],
        "ansatte i det offentlige": ["offentligt ansat", "tjenestemænd", "offentlig myndighed"],
        "forskere og nøglemedarbejdere": ["forsker", "nøglemedarbejder", "forskerskatteordning"],
        "kunstnere og sportsudøvere": ["kunstner", "sportsudøver", "atlet"],
        "pensionister": ["pensionist", "pension", "efterløn"],
        "studerende": ["studerende", "elev", "lærling"]
    }
    
    for chunk in chunks:
        # Lav en kopi af chunken
        updated_chunk = chunk.copy()
        content = chunk.get("content", "").lower()
        
        # Sikr at metadata indeholder de nødvendige felter
        if "metadata" not in updated_chunk:
            updated_chunk["metadata"] = {}
        
        if "legal_exceptions" not in updated_chunk["metadata"]:
            updated_chunk["metadata"]["legal_exceptions"] = []
        
        if "affected_groups" not in updated_chunk["metadata"]:
            updated_chunk["metadata"]["affected_groups"] = []
        
        # 1. Find undtagelser baseret på generelle mønstre
        for pattern in exception_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                exception = match.group(0).strip()
                if exception and len(exception) > 10:  # Undgå for korte udtryk
                    # Tjek om undtagelsen allerede er registreret (eller en variant)
                    already_exists = False
                    for existing in updated_chunk["metadata"]["legal_exceptions"]:
                        if exception.lower() in existing.lower() or existing.lower() in exception.lower():
                            already_exists = True
                            break
                    
                    if not already_exists:
                        updated_chunk["metadata"]["legal_exceptions"].append(exception)
        
        # 2. Find specifikke skattemæssige undtagelser
        for pattern, exceptions in specific_exceptions.items():
            if re.search(pattern, content, re.IGNORECASE):
                for exception in exceptions:
                    if exception not in updated_chunk["metadata"]["legal_exceptions"]:
                        updated_chunk["metadata"]["legal_exceptions"].append(exception)
        
        # 3. Find persongrupper der kan være omfattet
        for group, keywords in person_groups.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', content):
                    if group not in updated_chunk["metadata"]["affected_groups"]:
                        updated_chunk["metadata"]["affected_groups"].append(group)
                    break  # Kun tilføj gruppen én gang
        
        updated_chunks.append(updated_chunk)
    
    return updated_chunks

def normalize_paragraph_formats(chunks, context_summary):
    """Sikrer konsistent formatering af paragraffer og stykker baseret på kontekst"""
    normalized_chunks = []
    
    # Udpak korrekt formatering fra kontekst
    normalized_formats = {}
    
    if "summary" in context_summary and "document_structure" in context_summary["summary"]:
        document_structure = context_summary["summary"]["document_structure"]
        
        # For paragraffer
        for para, info in document_structure.items():
            # Normalisér til lowercase uden mellemrum for sammenligning
            key = re.sub(r'\s+', '', para.lower())
            normalized_formats[key] = para  # Gem originalt format
            
            # For stykker
            if isinstance(info, list):
                for stykke in info:
                    stykke_key = re.sub(r'\s+', '', stykke.lower())
                    normalized_formats[stykke_key] = stykke
    
    # Normaliser formatet for hver chunk
    for chunk in chunks:
        # Lav en kopi af chunken
        normalized_chunk = chunk.copy()
        
        # Sikr at metadata eksisterer
        if "metadata" not in normalized_chunk:
            normalized_chunk["metadata"] = {}
        
        # Normalisér paragraf
        if "paragraph" in normalized_chunk["metadata"] and normalized_chunk["metadata"]["paragraph"]:
            para = normalized_chunk["metadata"]["paragraph"]
            para_key = re.sub(r'\s+', '', para.lower())
            
            if para_key in normalized_formats:
                normalized_chunk["metadata"]["paragraph"] = normalized_formats[para_key]
        
        # Normalisér stykke
        if "stykke" in normalized_chunk["metadata"] and normalized_chunk["metadata"]["stykke"]:
            stykke = normalized_chunk["metadata"]["stykke"]
            stykke_key = re.sub(r'\s+', '', stykke.lower())
            
            if stykke_key in normalized_formats:
                normalized_chunk["metadata"]["stykke"] = normalized_formats[stykke_key]
        
        normalized_chunks.append(normalized_chunk)
    
    return normalized_chunks

def validate_preserved_notes(chunks, preserved_content):
    """Validerer at noter er bevaret komplet i chunks i forhold til deres originaltekst"""
    results = {
        "trunkerede_noter": [],
        "modificerede_noter": [],
        "komplette_noter": 0,
        "manglende_noter": []
    }
    
    if not preserved_content or "notes" not in preserved_content:
        return results
    
    preserved_notes = preserved_content["notes"]
    found_note_numbers = set()
    
    for note_num, original_content in preserved_notes.items():
        # Find alle chunks for denne note
        note_chunks = [
            c for c in chunks 
            if c["metadata"].get("is_note", False) and str(c["metadata"].get("note_number", "")) == str(note_num)
        ]
        
        if not note_chunks:
            results["manglende_noter"].append(note_num)
            continue
        
        found_note_numbers.add(note_num)
        
        # Sammenlign indhold
        combined_content = " ".join([
            re.sub(r'\[NOTE:\d+\]\s*', '', c["content"]) for c in note_chunks
        ]).strip()
        
        original_content = original_content.strip()
        
        # Tjek om alt væsentligt indhold er bevaret
        if len(combined_content) < len(original_content) * 0.9:  # Tillad små forskelle
            results["trunkerede_noter"].append(note_num)
        elif len(combined_content) > len(original_content) * 1.5:  # Muligvis fordoblet indhold
            results["modificerede_noter"].append(note_num)
        else:
            # Tjek for specifikke nøglefraser for visse noter
            if note_num == "795" and not all(key_phrase.lower() in combined_content.lower() 
                                           for key_phrase in ["grænsegængere", "kildeskattelovens §§ 5 A-5 D"]):
                results["modificerede_noter"].append(note_num)
            else:
                results["komplette_noter"] += 1
    
    # Tjek for noter i chunks som ikke var i det oprindelige indhold
    for chunk in chunks:
        if chunk["metadata"].get("is_note", False):
            note_num = str(chunk["metadata"].get("note_number", ""))
            if note_num and note_num not in found_note_numbers and note_num not in preserved_notes:
                results["modificerede_noter"].append(note_num)
    
    return results

def process_with_improved_methods(chunks, context_summary, preserved_content=None):
    """Kører alle forbedrede metoder for at sikre juridisk korrekthed og konsistens"""
    # Initialiser statistik
    stats = {
        "initial_validation": {"overall_status": "unknown"},
        "improved_validation": {"overall_status": "unknown"},
        "note_validation": {"komplette_noter": 0},
        "improvements": {
            "chunks_before": len(chunks),
            "chunks_after": len(chunks),
        }
    }
    
    # Sikkerhedskopi af chunks
    improved_chunks = chunks
    
    # Kør validering først for at identificere problemer
    validation_results = None
    try:
        validation_results = validate_chunks(chunks, context_summary)
        stats["initial_validation"] = validation_results
    except Exception as e:
        print(f"Validering fejlede: {str(e)}")
        validation_results = {
            "missing_paragraphs": [],
            "overall_status": "error",
            "error_message": str(e)
        }
        stats["initial_validation"] = validation_results
    
    try:
        # 1. Normalisér paragraf- og stykkeformater
        improved_chunks = normalize_paragraph_formats(improved_chunks, context_summary)
        
        # 2. Ekstraher juridiske undtagelser og specialgrupper
        improved_chunks = extract_legal_exceptions_from_content(improved_chunks)
        
        # 3. Reparer manglende paragraffer og stykker
        improved_chunks = repair_missing_paragraphs(improved_chunks, context_summary, validation_results, preserved_content)
        
        # 4. Fjern redundante chunks
        from utils.optimization import optimize_chunks
        improved_chunks = optimize_chunks(improved_chunks)
        
        # 5. Balancér chunk-størrelser
        from utils.optimization import merge_small_chunks, split_large_chunks
        improved_chunks = merge_small_chunks(improved_chunks, min_size=200, target_size=1000)
        improved_chunks = split_large_chunks(improved_chunks, max_size=2000)
        
        # 6. Lav en ny validering for at tjekke resultatet
        try:
            final_validation = validate_chunks(improved_chunks, context_summary)
            stats["improved_validation"] = final_validation
        except Exception as e:
            print(f"Final validering fejlede: {str(e)}")
            stats["improved_validation"] = {
                "overall_status": "error",
                "error_message": str(e)
            }
        
        # 7. Validér at noter er bevaret komplet
        try:
            note_validation = validate_preserved_notes(improved_chunks, preserved_content)
            stats["note_validation"] = note_validation
        except Exception as e:
            print(f"Note validering fejlede: {str(e)}")
            stats["note_validation"] = {
                "error_message": str(e)
            }
        
    except Exception as e:
        print(f"Forbedring fejlede: {str(e)}")
        # Ved fejl, returner de originale chunks
        return chunks, {"error": str(e)}
    
    # Opdater statistik
    stats["improvements"]["chunks_after"] = len(improved_chunks)
    
    return improved_chunks, stats