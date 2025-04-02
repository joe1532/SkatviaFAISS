import re
import streamlit as st
import numpy as np

def segment_text_for_processing(text, max_segment_length=30000):
    """
    Opdeler tekst i segmenter til indeksering med forbedret juridisk hensyn.
    
    Args:
        text: Tekst der skal segmenteres
        max_segment_length: Maksimal længde på et segment
        
    Returns:
        Liste af tekstsegmenter og bevarede indholdselementer
    """
    # Log original tekstlængde
    st.write(f"Original tekstlængde: {len(text)} tegn")
    st.write(f"Maksimal segmentlængde: {max_segment_length} tegn")
    
    segments = []
    
    # Oversigt over indhold som skal bevares intakt
    preserved_content = {
        "notes": {},
        "paragraphs": {},
        "sections": {},
        "examples": {}
    }
    
    # 1. Del efter "NOTER:" mærket hvis det findes
    parts = re.split(r'(NOTER:|\nNoter\n)', text, 1)
    main_text = parts[0]
    notes_text = "".join(parts[1:]) if len(parts) > 1 else ""
    
    # Hvis det ikke lykkedes at finde noter-sektionen, prøv igen med regulære udtryk
    if not notes_text:
        # Søg efter første trecifrede tal efterfulgt af tekst - det er ofte starten på noterne
        note_match = re.search(r'\n(\d{3})\s+', text[len(main_text)//2:])
        if note_match:
            notes_start_pos = len(main_text)//2 + note_match.start()
            main_text = text[:notes_start_pos]
            notes_text = text[notes_start_pos:]
    
    # Gem referencer til note-tekst
    note_pattern = r'(?:\n|\[NOTE:)(\d{3})(?:\]|\s+)([^\n]+(?:\n(?!\d{3})[^\n]+)*)'
    note_matches = re.finditer(note_pattern, notes_text, re.DOTALL)
    
    for match in note_matches:
        note_num = match.group(1)
        note_content = match.group(2).strip()
        preserved_content["notes"][note_num] = note_content
    
    # 2. Del hovedtekst ved juridisk betydningsfulde grænser
    
    # A. Prøv først at finde afsnit baseret på juridisk vejlednings-struktur (C.F.X.X.X)
    jv_section_pattern = r'(C\.F\.\d+\.\d+\.\d+\s+.+?)(?=C\.F\.\d+\.\d+\.\d+|$)'
    jv_matches = list(re.finditer(jv_section_pattern, main_text, re.DOTALL))
    
    if jv_matches:
        # Den Juridiske Vejledning-struktur
        for match in jv_matches:
            segment = match.group(1)
            segments.append(segment)
            
            # Uddrag afsnits-ID
            section_id_match = re.search(r'(C\.F\.\d+\.\d+\.\d+)', segment)
            if section_id_match:
                section_id = section_id_match.group(1)
                preserved_content["sections"][section_id] = segment
    else:
        # B. Prøv at finde paragrafgrænser hvis det ikke er JV
        paragraph_pattern = r'(§\s+\d+[A-Za-z]?|Kapitel\s+\d+|Afsnit\s+\d+)'
        paragraphs = re.split(paragraph_pattern, main_text)
        
        current_segment = ""
        for i in range(0, len(paragraphs)-1, 2):
            if i+1 < len(paragraphs):
                paragraph_marker = paragraphs[i]
                paragraph_content = paragraphs[i+1]
                full_paragraph = paragraph_marker + paragraph_content
                
                # Bevar original paragraftekst
                section_id = paragraph_marker.strip()
                preserved_content["paragraphs"][section_id] = full_paragraph
                
                # Hvis current_segment ville blive for stort, gem det og start en ny
                if len(current_segment + full_paragraph) > max_segment_length:
                    if current_segment:
                        segments.append(current_segment)
                    
                    # Hvis selve paragraffen er for stor, opdel den
                    if len(full_paragraph) > max_segment_length:
                        st.warning(f"Paragraf '{section_id}' er for stor ({len(full_paragraph)} tegn). Opdeler i mindre dele.")
                        # Del i sætninger eller subsections
                        sub_parts = split_with_juridical_awareness(full_paragraph)
                        
                        # Maksimalt segment-size
                        current_sub_segment = ""
                        for part in sub_parts:
                            if len(current_sub_segment + part) > max_segment_length:
                                if current_sub_segment:
                                    segments.append(current_sub_segment)
                                current_sub_segment = part
                            else:
                                current_sub_segment += part
                        
                        if current_sub_segment:
                            segments.append(current_sub_segment)
                    else:
                        current_segment = full_paragraph
                else:
                    current_segment += full_paragraph
        
        # Tilføj sidste segment
        if current_segment:
            segments.append(current_segment)
        
        # C. Hvis ingen paragraffer blev fundet, del ved semantiske grænser
        if not segments:
            segments = split_with_juridical_awareness(main_text, max_segment_length)
    
    # 3. Udpak eksempler
    example_pattern = r'(Eksempel(?:\s+\d+)?:(?:.*?)(?=\n\n\w|Eksempel(?:\s+\d+)?:|$))'
    for segment in ' '.join(segments):
        for match in re.finditer(example_pattern, segment, re.DOTALL):
            example_text = match.group(1).strip()
            # Generer et unikt ID for eksemplet
            example_id = f"eks_{len(preserved_content['examples'])+1}"
            preserved_content["examples"][example_id] = example_text
    
    # 4. Behandl noter som separate segmenter
    note_segments = []
    
    # Opdel noterne i mindre chunks også
    for note_num, note_content in preserved_content["notes"].items():
        full_note = f"[NOTE:{note_num}] {note_content}"
        if len(full_note) > max_segment_length:
            st.warning(f"Note {note_num} er for stor ({len(full_note)} tegn). Opdeler i mindre dele.")
            
            # Del noten op
            note_parts = split_with_juridical_awareness(full_note, max_segment_length // 2)
            for i, part in enumerate(note_parts):
                note_segments.append(f"[NOTE:{note_num} del {i+1}] {part}")
        else:
            note_segments.append(full_note)
    
    # Hvis ingen notesegmenter men vi har notes_text, forsøg at opdele notes_text
    if not note_segments and notes_text:
        # Forsøg at opdele notes_text baseret på note-numre
        note_segments = split_notes_text(notes_text, max_segment_length)
    
    # Kombiner hoved-segmenter og note-segmenter
    segments.extend(note_segments)
    
    # Opdel eventuelle resterende store segmenter
    final_segments = []
    for segment in segments:
        if len(segment) > max_segment_length:
            st.warning(f"Fandt et segment på {len(segment)} tegn, som er større end max ({max_segment_length}). Opdeler det yderligere.")
            # Del i mindre stykker
            for i in range(0, len(segment), max_segment_length // 2):
                part = segment[i:i + max_segment_length // 2]
                final_segments.append(part)
        else:
            final_segments.append(segment)
    
    # Log statistik
    stats = {
        "main_segments": len(segments) - len(note_segments),
        "note_segments": len(note_segments),
        "total_segments": len(final_segments),
        "preserved_notes": len(preserved_content["notes"]),
        "preserved_paragraphs": len(preserved_content["paragraphs"]),
        "preserved_sections": len(preserved_content["sections"]),
        "preserved_examples": len(preserved_content["examples"])
    }
    
    # Log information om segmenter
    st.write(f"Segmenteret tekst i {len(final_segments)} dele:")
    for i, segment in enumerate(final_segments):
        st.write(f"Segment {i+1}: {len(segment)} tegn")
    
    return final_segments, preserved_content, stats

def split_with_juridical_awareness(text, max_length=15000):
    """
    Deler tekst i semantiske segmenter med hensyn til juridisk struktur.
    Denne funktion sikrer, at vi ikke deler midt i juridiske ræsonnementer eller definitioner.
    
    Args:
        text: Tekst der skal opdeles
        max_length: Maksimal længde for hvert segment
        
    Returns:
        Liste af segmenter
    """
    segments = []
    
    # 1. Prøv først at dele ved eksplicitte sektionsmarkører
    section_markers = [
        r'\n\s*\n[A-Z][a-zA-ZæøåÆØÅ\s]+\n',  # Overskrifter
        r'\n\s*\n(Betingelser|Forudsætninger|Undtagelser|Hovedregel|Praksis|Eksempel|Se også|Bemærk)',
        r'\n\s*\n\d+\.\s+[A-ZÆØÅ]',  # Nummererede afsnit
        r'\n\s*\n[•\-]\s+'  # Punkter
    ]
    
    # Find alle potentielle breakpoints
    breaks = []
    for marker in section_markers:
        for match in re.finditer(marker, text):
            breaks.append(match.start())
    
    # Sortér breaks og fjern duplikater
    breaks = sorted(set(breaks))
    
    # Hvis ingen breaks blev fundet eller de ikke giver passende segmenter
    if not breaks or (breaks and breaks[0] > max_length):
        # 2. Del ved afsnit
        paragraphs = text.split('\n\n')
        
        current_segment = ""
        for para in paragraphs:
            if not para.strip():
                continue
                
            if len(current_segment + para + "\n\n") <= max_length:
                current_segment += para + "\n\n"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                    
                # Hvis paragraffen selv er for stor, del ved sætninger
                if len(para) > max_length:
                    sentences = split_into_sentences(para)
                    current_segment = ""
                    
                    for sentence in sentences:
                        if len(current_segment + sentence) <= max_length:
                            current_segment += sentence
                        else:
                            if current_segment:
                                segments.append(current_segment.strip())
                            current_segment = sentence
                else:
                    current_segment = para + "\n\n"
        
        if current_segment:
            segments.append(current_segment.strip())
    else:
        # Brug de fundne breaks til at dele teksten
        current_pos = 0
        for break_pos in breaks:
            # Hvis dette segment ville blive for stort, del det yderligere
            if break_pos - current_pos > max_length:
                # Del dette segment ved afsnit eller sætninger
                subsegment = text[current_pos:break_pos]
                subsegments = []
                
                # Find afsnit inden for dette segment
                subparagraphs = subsegment.split('\n\n')
                
                current_subsegment = ""
                for para in subparagraphs:
                    if len(current_subsegment + para + "\n\n") <= max_length:
                        current_subsegment += para + "\n\n"
                    else:
                        if current_subsegment:
                            subsegments.append(current_subsegment.strip())
                        current_subsegment = para + "\n\n"
                
                if current_subsegment:
                    subsegments.append(current_subsegment.strip())
                
                segments.extend(subsegments)
            else:
                # Dette segment er passende størrelse
                segment = text[current_pos:break_pos].strip()
                if segment:
                    segments.append(segment)
            
            current_pos = break_pos
        
        # Tilføj sidste segment
        if current_pos < len(text):
            final_segment = text[current_pos:].strip()
            if final_segment:
                if len(final_segment) <= max_length:
                    segments.append(final_segment)
                else:
                    # Del sidste segment hvis det er for stort
                    final_subsegments = []
                    subparagraphs = final_segment.split('\n\n')
                    
                    current_subsegment = ""
                    for para in subparagraphs:
                        if len(current_subsegment + para + "\n\n") <= max_length:
                            current_subsegment += para + "\n\n"
                        else:
                            if current_subsegment:
                                final_subsegments.append(current_subsegment.strip())
                            current_subsegment = para + "\n\n"
                    
                    if current_subsegment:
                        final_subsegments.append(current_subsegment.strip())
                    
                    segments.extend(final_subsegments)
    
    # Hvis vi stadig ikke har segmenter eller har for store segmenter
    final_segments = []
    for segment in segments:
        if len(segment) > max_length:
            # Del dette segment ved sætninger
            sentences = split_into_sentences(segment)
            
            current_segment = ""
            for sentence in sentences:
                if len(current_segment + sentence) <= max_length:
                    current_segment += sentence
                else:
                    if current_segment:
                        final_segments.append(current_segment.strip())
                    current_segment = sentence
            
            if current_segment:
                final_segments.append(current_segment.strip())
        else:
            final_segments.append(segment)
    
    return final_segments if final_segments else [text]

def split_into_sentences(text):
    """
    Opdeler tekst i sætninger med hensyn til juridiske forkortelser.
    
    Args:
        text: Tekst der skal opdeles i sætninger
        
    Returns:
        Liste af sætninger
    """
    # Almindelige juridiske forkortelser der indeholder punktum
    abbreviations = [
        r'jf\.', r'bl\.a\.', r'f\.eks\.', r'pkt\.', r'nr\.', r'stk\.', 
        r'ca\.', r'evt\.', r'osv\.', r'mv\.', r'inkl\.', r'ekskl\.',
        r'hhv\.', r'vedr\.', r'afd\.', r'div\.', r'pga\.',
        r'SKM\.', r'TfS\.', r'RR\.'
    ]
    
    # Erstat forkortelser midlertidigt
    for abbr in abbreviations:
        text = re.sub(abbr, abbr.replace('.', '<DOT>'), text)
    
    # Del ved sætningsgrænser
    # Bemærk: Vi ser efter punktum efterfulgt af mellemrum og stort bogstav
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÆØÅ])', text)
    
    # Gendan forkortelser
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    
    return sentences

def split_notes_text(notes_text, max_length=15000):
    """
    Specielt designet til at opdele noter sektion optimalt.
    
    Args:
        notes_text: Tekst med noter
        max_length: Maksimal længde for et segment
        
    Returns:
        Liste af note-segmenter
    """
    segments = []
    
    # Del ved notenumre
    note_pattern = r'(\n\d{3}\s+|\[NOTE:\d{3}\])'
    parts = re.split(note_pattern, notes_text)
    
    current_segment = ""
    current_marker = ""
    
    # Vi får skiftevis markører og indhold
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Dette er en markør (notenummer)
        if re.match(r'(\n\d{3}\s+|\[NOTE:\d{3}\])', part):
            current_marker = part
            continue
        
        # Dette er noteindhold
        if current_marker:
            note_content = current_marker + part
            
            # Hvis denne note ville gøre segmentet for stort, start et nyt segment
            if len(current_segment + note_content) > max_length:
                if current_segment:
                    segments.append(current_segment)
                
                # Hvis selve noten er for stor, del den yderligere
                if len(note_content) > max_length:
                    note_num = re.search(r'\d{3}', current_marker).group(0)
                    note_parts = split_with_juridical_awareness(note_content, max_length // 2)
                    
                    for j, note_part in enumerate(note_parts):
                        segments.append(f"[NOTE:{note_num} del {j+1}/{len(note_parts)}] {note_part}")
                else:
                    current_segment = note_content
            else:
                current_segment += note_content
            
            current_marker = ""
    
    # Tilføj sidste segment
    if current_segment:
        segments.append(current_segment)
    
    return segments

def normalize_case_references(chunks):
    """
    Normaliserer domsreferencer til standardformat på tværs af alle chunks.
    
    Args:
        chunks: Liste af chunks at behandle
        
    Returns:
        Liste af chunks med normaliserede referencer
    """
    for chunk in chunks:
        if "metadata" not in chunk:
            continue
            
        metadata = chunk["metadata"]
        if "case_references" not in metadata or not metadata["case_references"]:
            # Ingen referencer at normalisere
            metadata["normalized_case_references"] = []
            continue
        
        normalized_refs = []
        for ref in metadata["case_references"]:
            # Standardformat: PREFIX.YEAR.NUMBER.INSTANCE
            
            # SKM-format: SKM2020.123.LSR -> SKM.2020.123.LSR
            skm_match = re.search(r'SKM[.\s]?(\d{4})[.\s]?(\d+)[.\s]?([A-Z]+)', ref)
            if skm_match:
                normalized = f"SKM.{skm_match.group(1)}.{skm_match.group(2)}.{skm_match.group(3)}"
                normalized_refs.append(normalized)
                continue
            
            # TfS-format: TfS 2020, 123 H -> TfS.2020.123.H
            tfs_match = re.search(r'TfS[.\s]?(\d{4})[,.\s]?(\d+)(?:[.\s]?([A-Z]+))?', ref)
            if tfs_match:
                instance = tfs_match.group(3) or ''
                normalized = f"TfS.{tfs_match.group(1)}.{tfs_match.group(2)}"
                if instance:
                    normalized += f".{instance}"
                normalized_refs.append(normalized)
                continue
            
            # U-format (Ugeskrift for Retsvæsen): U 2020.123 H -> U.2020.123.H
            u_match = re.search(r'U[.\s]?(\d{4})[.\s]?(\d+)(?:[.\s]?([A-Z]+))?', ref)
            if u_match:
                instance = u_match.group(3) or ''
                normalized = f"U.{u_match.group(1)}.{u_match.group(2)}"
                if instance:
                    normalized += f".{instance}"
                normalized_refs.append(normalized)
                continue
            
            # Hvis ingen match, behold originalen
            normalized_refs.append(ref)
        
        # Fjern duplikater
        normalized_refs = list(set(normalized_refs))
        metadata["normalized_case_references"] = normalized_refs
    
    return chunks

def extract_sections_from_text(text):
    """
    Ekstraherer sektioner og struktur fra juridisk tekst.
    
    Args:
        text: Juridisk tekst at analysere
        
    Returns:
        Dict med sektioner og deres struktur
    """
    sections = {}
    structure = {"hierarchy": {}, "order": []}
    
    # Prøv først JV-struktur
    jv_pattern = r'(C\.F\.\d+\.\d+\.\d+)\s+(.+?)(?=C\.F\.\d+\.\d+\.\d+|$)'
    jv_matches = list(re.finditer(jv_pattern, text, re.DOTALL))
    
    if jv_matches:
        for match in jv_matches:
            section_id = match.group(1)
            section_content = match.group(0)
            section_title = match.group(2).strip().split('\n')[0]
            
            sections[section_id] = {
                "content": section_content,
                "title": section_title
            }
            
            structure["order"].append(section_id)
            
            # Opbyg hierarki
            # C.F.4.2.1 -> parent: C.F.4.2
            parts = section_id.split('.')
            if len(parts) >= 4:
                parent_id = f"{parts[0]}.{parts[1]}.{parts[2]}"
                if parent_id not in structure["hierarchy"]:
                    structure["hierarchy"][parent_id] = []
                structure["hierarchy"][parent_id].append(section_id)
    else:
        # Prøv paragraph struktur
        para_pattern = r'(§\s+\d+[a-z]?)\s+(.+?)(?=§\s+\d+|$)'
        para_matches = list(re.finditer(para_pattern, text, re.DOTALL))
        
        if para_matches:
            for match in para_matches:
                section_id = match.group(1)
                section_content = match.group(0)
                
                # Forsøg at udtrække titel baseret på første linje
                first_line = match.group(2).strip().split('\n')[0]
                section_title = first_line if len(first_line) < 100 else ""
                
                sections[section_id] = {
                    "content": section_content,
                    "title": section_title
                }
                
                structure["order"].append(section_id)
                
                # For paragraffer, forsøg at finde stykker
                stk_pattern = r'([Ss]tk\.\s+\d+)[.\s]'
                stk_matches = re.finditer(stk_pattern, section_content)
                
                stykker = []
                for stk_match in stk_matches:
                    stykker.append(stk_match.group(1))
                
                if stykker:
                    structure["hierarchy"][section_id] = stykker
    
    return {"sections": sections, "structure": structure}