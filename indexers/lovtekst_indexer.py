# indexers/lovtekst_indexer.py
import re
import streamlit as st
import json
from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils, indexing
from utils.optimization import cached_call_gpt4o, process_segments_parallel, optimize_chunks

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Lovtekst-indekserer"
        self.description = "Specialiseret indeksering af lovbekendtgørelser"
        
        self.document_types = {
            "ligningsloven": {
                "display_name": "Ligningsloven",
                "template_name": "ligningslov_template",
                "note_pattern": r'\d{3}',
                "paragraph_pattern": r'§\s*\d+\s*[A-Za-z]?',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
            "personskatteloven": {
                "display_name": "Personskatteloven",
                "template_name": "personskattelov_template",
                "note_pattern": r'\d{2,3}',
                "paragraph_pattern": r'§\s*\d+\s*[A-Za-z]?',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
        }
    
    def display_settings(self, st):
        """Viser indstillinger specifikt for lovtekster"""
        st.session_state.has_numbered_notes = st.checkbox(
            "Dokumentet indeholder noter markeret med numre (f.eks. 794, 795)", 
            value=True
        )
        st.session_state.has_case_references = st.checkbox(
            "Dokumentet indeholder domme og afgørelser (SKM, TfS, osv.)", 
            value=True
        )
        
        return "lovtekst"  # Returnerer bare "lovtekst" som doc_type_key uden yderligere undertyper
    
    def process_document(self, text, doc_id, options):
        """
        Processer lovtekst-dokument med indeksering
        """
        # 1. Preprocessering
        processed_text, text_sections = pdf_utils.preprocess_legal_text(text)
        st.session_state.original_text_sections = text_sections
        st.session_state.original_text = text
        
        # 2. Segmentering
        segments, preserved_content, segment_stats = text_analysis.segment_text_for_processing(
            processed_text, max_segment_length=options.get("max_text_length", 30000)
        )
        st.session_state.preserved_content = preserved_content
        processing_stats = segment_stats
        
        # Hent doc_type_key fra options
        doc_type_key = options.get("doc_type_key")
        if not doc_type_key:
            doc_type_key = "lovtekst"  # Fallback til generisk lovtekst
        
        # 3. Kontekstanalyse med caching
        with st.spinner("Analyserer dokumentets struktur og indhold..."):
            context_prompt = self.get_context_prompt_template(doc_type_key)
            context_prompt_with_text = context_prompt + "\n\nDokument:\n" + segments[0]
            
            context_summary = cached_call_gpt4o(
                context_prompt_with_text,
                model=options.get("model", "gpt-4o")
            )
            if not context_summary:
                st.error("Kunne ikke generere kontekstopsummering. Prøv igen.")
                return None, None
        
        # 4. Chunking med parallelisering
        with st.spinner("Opdeler dokumentet i meningsfulde chunks..."):
            chunks = process_segments_parallel(
                segments, 
                doc_type_key, 
                context_summary, 
                doc_id, 
                options,
                get_template_func=self.get_indexing_prompt_template
            )
        
        # 5. Kør optimering på alle chunks EFTER de er indsamlet
        if chunks:
            chunks = optimize_chunks(chunks)
        
        return chunks, context_summary
    
    def get_context_prompt_template(self, doc_type_key):
        """Henter kontekstprompt skabelonen baseret på dokumenttype."""
        return """
        Du er en ekspert i dansk skatteret. Analyser denne lovtekst og opbyg en forståelse af dens struktur og indhold.
        
        RETURNER DIN SVAR SOM JSON.
        
        Returner en JSON-opsummering med:
        - Dokumentets struktur (paragraffer og stykker)
        - Hovedtemaer og nøglebegreber i dokumentet
        - Juridiske undtagelser og specialtilfælde
        - Noter og fortolkningsbidrag hvis de findes
        
        Format:
        {
          "document_id": "unik_id_for_dokumentet",
          "document_type": "lovtekst", 
          "version_date": "YYYY-MM-DD",
          "summary": {
            "main_themes": ["tema1", "tema2"],
            "key_concepts": ["nøgleord1", "nøgleord2"],
            "document_structure": {
              "§ 1": ["Stk. 1", "Stk. 2"],
              "§ 2": ["Stk. 1"]
            },
            "section_titles": {"§ 1": "Titel for paragraf 1"},
            "notes_overview": {
              "794": {
                "text": "Første del af noten...",
                "references": ["§ 33 A"],
                "key_legal_exceptions": ["Undtagelsesregel 1"]
              }
            },
            "legal_exceptions": [
              {
                "rule": "Hovedregel",
                "exception": "Undtagelse",
                "source": "§ X, Stk. Y"
              }
            ]
          }
        }
        """
    
    def get_indexing_prompt_template(self, doc_type_key, context_summary, doc_id, section_number):
        """Henter indekseringsprompt skabelonen baseret på dokumenttype."""
        return f"""
        Du er en ekspert i dansk skatteret der skal indeksere lovtekst.
        Din opgave er at opdele denne tekst i chunks. Hvert chunk skal være en logisk indholdsdel.
        
        Du har fået denne kontekst:
        {json.dumps(context_summary, ensure_ascii=False)}
        
        Jeg viser dig nu sektion {section_number} af dokumentet, som du skal opdele i chunks.
        
        Du SKAL følge disse regler:
        1. BEVAR DEN KOMPLETTE, UÆNDREDE tekst i hvert chunk. Lav ALDRIG opsummeringer eller parafraseringer.
        2. Opdel teksten logisk ved paragraffer eller naturlige brudpunkter.
        3. Opdel ALDRIG midt i en sætning.
        4. Chunks må ikke være for lange eller korte. Aim for hele logiske afsnit.
        5. Du SKAL returnere resultatet som et JSON-objekt med en top-level ARRAY kaldet "chunks".
        
        VIGTIGT: Returneringsformatet SKAL være nøjagtigt dette:
        {{
          "chunks": [
            {{
              "content": "NØJAGTIG tekst fra kilden uden ændringer",
              "metadata": {{
                "doc_id": "{doc_id}",
                "paragraph": "§ X",
                "stykke": "Stk. Y",
                "concepts": ["nøgleord1", "nøgleord2", "nøgleord3"],
                "law_references": ["Ligningslovens § 33 A, stk. 1"],
                "is_note": false,
                "note_number": "",
                "theme": "tema",
                "subtheme": "undertema",
                "status": "gældende",
                "affected_groups": []
              }}
            }},
            {{
              "content": "NØJAGTIG tekst fra anden chunk",
              "metadata": {{ ... }}
            }}
          ]
        }}
        
        RETURNER DIN SVAR SOM JSON med strukturen som er angivet ovenfor. Det er meget vigtigt at der er en "chunks" array på øverste niveau.
        """