import io
import re
import streamlit as st
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    """
    Ekstraherer tekst fra en PDF-fil.
    
    Args:
        pdf_file: PDF-fil objekt
        
    Returns:
        Ekstraheret tekst
    """
    pdf_reader = PdfReader(pdf_file)
    full_text = ""
    page_texts = []
    
    # Ekstrahér tekst side for side
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_texts.append(page_text)
            
    # Kombiner sidetekster til fuld tekst
    full_text = "\n\n".join(page_texts)
    
    # Gem statistik
    stats = {
        "pdf_pages": len(pdf_reader.pages),
        "raw_text_length": len(full_text)
    }
    
    return full_text, stats  # Returnerer KUN teksten og statistik, ikke et tuple

def preprocess_legal_text(text):
    """
    Preprocesserer juridisk tekst fra PDF.
    
    Args:
        text: Rå tekst fra PDF
        
    Returns:
        Tuple med preprocesseret tekst og opdelt tekst i hovedtekst og noter
    """
    # Fjern PDF konverteringsproblemer
    processed_text = re.sub(r'([a-zA-Z])-\n([a-zA-Z])', r'\1\2', text)  # Hyphens
    processed_text = re.sub(r'\n+', '\n', processed_text)  # Multiple newlines
    
    # Normalisér mellemrum, men bevar afsnit
    processed_text = re.sub(r' +', ' ', processed_text)
    
    # Normalisér paragraftegn og stykke - bevar konsistent formatering
    processed_text = re.sub(r'§\s*(\d+[a-zA-Z]?)', r'§ \1', processed_text)
    processed_text = re.sub(r'[sS]tk\.\s*(\d+)', r'Stk. \1', processed_text)
    
    # Del teksten i hovedtekst og noter
    main_text_and_notes = split_into_main_text_and_notes(processed_text)
    
    return processed_text, main_text_and_notes

def split_into_main_text_and_notes(text):
    """
    Opdeler teksten i hovedtekst og noter.
    
    Args:
        text: Tekst der skal opdeles
        
    Returns:
        Dictionary med hovedtekst og noter
    """
    # Forsøg at identificere notesektionen
    note_start_patterns = [
        r'\nNoter\n',
        r'\nNOTER:\n',
        r'\n\d{3}\s+?[§A-Za-z]'  # Første note (f.eks. "794 § 33 A er...")
    ]
    
    sections = {}
    main_text = text
    
    for pattern in note_start_patterns:
        parts = re.split(pattern, text, 1)
        if len(parts) > 1:
            main_text = parts[0]
            notes_text = parts[1] if len(parts) > 1 else ""
            
            # Forsøg at identificere individuelle noter
            notes = extract_individual_notes(notes_text)
            
            sections["main_text"] = main_text
            sections["notes"] = notes
            break
    
    if "notes" not in sections:
        sections["main_text"] = text
        sections["notes"] = []
    
    return sections

def extract_individual_notes(notes_text):
    """
    Ekstraherer individuelle noter fra noteteksten.
    
    Args:
        notes_text: Notetekst
        
    Returns:
        Liste af noter med nummer og indhold
    """
    notes = []
    
    # Match noter markeret med NOTE-tag eller med start på 3 cifre
    note_pattern = r'(?:\[NOTE:(\d{3})\]|^(\d{3})|\n(\d{3}))\s*(.*?)(?=\n\d{3}|\[NOTE:\d{3}\]|$)'
    matches = re.finditer(note_pattern, notes_text, re.DOTALL)
    
    for match in matches:
        note_num = match.group(1) or match.group(2) or match.group(3)
        note_content = match.group(4).strip()
        
        if note_num and note_content:
            notes.append({
                "number": note_num,
                "content": f"[NOTE:{note_num}] {note_content}"
            })
    
    return notes