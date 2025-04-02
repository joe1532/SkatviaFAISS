# indexers/generisk_indexer.py
import re
import streamlit as st
import json
from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils, indexing
from utils.optimization import cached_call_gpt4o, process_segments_parallel

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Generisk indekserer"
        self.description = "Generisk indeksering for dokumenter der ikke falder i andre kategorier"
    
    def display_settings(self, st):
        """Viser indstillinger for generisk indeksering"""
        doc_type_key = st.selectbox(
            "Dokumenttype:",
            ["generisk", "lovforslag", "betænkning", "andet"],
            format_func=lambda x: x.capitalize()
        )
        
        # Grundlæggende indstillinger
        st.session_state.preserve_paragraphs = st.checkbox(
            "Bevar paragrafstruktur", 
            value=True,
            help="Bevar paragrafstruktur hvis dokumentet indeholder paragraffer"
        )
        
        st.session_state.detect_references = st.checkbox(
            "Detekter eksterne referencer", 
            value=True,
            help="Find og normaliser referencer til domme, afgørelser, og andre dokumenter"
        )
        
        return doc_type_key
    
    def process_document(self, text, doc_id, options):
        """
        Processer dokument med generisk indeksering
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
        
        # 3. Kontekstanalyse
        with st.spinner("Analyserer dokumentets struktur og indhold..."):
            context_prompt = self.get_context_prompt_template(options.get("doc_type_key", "generisk"))
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
                options.get("doc_type_key", "generisk"), 
                context_summary, 
                doc_id, 
                options,
                get_template_func=self.get_indexing_prompt_template
            )
        
        # 5. Normalisér referencer hvis aktiveret
        if st.session_state.detect_references:
            chunks = text_analysis.normalize_case_references(chunks)
        
        return chunks, context_summary
    
    def get_context_prompt_template(self, doc_type_key):
        """Henter kontekstprompt skabelonen for generisk indeksering."""
        return """
        Du er en ekspert i dansk skatteret. Læs dette dokument og opbyg en forståelse af dets struktur og indhold.
        
        RETURNER DIN SVAR SOM JSON.
        
        Returner en JSON-opsummering med:
        - Dokumentets type og formål
        - Hovedtemaer og nøglebegreber
        - Struktur (kapitler, afsnit, punkter)
        - Referencer til love, paragraffer, og retskilder
        - Noter og fortolkningsbidrag hvis de findes
        
        Format:
        {
          "document_id": "unik_id_for_dokumentet",
          "document_type": "lovtekst/vejledning/andet", 
          "version_date": "YYYY-MM-DD",
          "summary": {
            "main_themes": ["tema1", "tema2"],
            "key_concepts": ["nøgleord1", "nøgleord2"],
            "document_structure": {
              "§ 1": ["Stk. 1", "Stk. 2"],
              "§ 2": ["Stk. 1"]
            },
            "section_titles": {"§ 1": "Titel for paragraf 1"},
            "law_references": ["Kildeskattelovens § 1", "Ligningslovens § 33 A"],
            "notes_overview": {
              "1": {
                "text": "Første del af noten...",
                "references": ["§ 1", "§ 2"],
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
        """Henter indekseringsprompt skabelonen for generisk indeksering."""
        return f"""
        Du er en ekspert i dansk skatteret der skal indeksere et dokument. 
        Du har fået denne kontekst: {json.dumps(context_summary, ensure_ascii=False)}.
        
        Dette er sektion {section_number} af dokumentet. Din opgave er at opdele denne sektion i semantisk meningsfulde chunks.
        
        Opdel teksten i semantisk meningsfulde chunks baseret på dokumentets struktur.
        CHUNK ALDRIG MIDT I EN SÆTNING. 
        
        VIGTIGSTE REGEL: Bevar den KOMPLETTE, UÆNDREDE tekst fra kilden. Lav ALDRIG opsummeringer eller parafraseringer.
        
        RETURNER DIN SVAR SOM JSON.
        
        Tildel følgende metadata til hvert chunk:
        1. Dokument-ID: "{doc_id}"
        2. Type (lovtekst, note, vejledning, osv.)
        3. Position (paragraf, stykke, afsnit, osv.) hvis relevant
        4. Nøgleord (maks 5 pr. chunk)
        5. Referencer til love/paragraffer
        6. Referencer til domme/afgørelser
        7. Normaliserede referencer
        8. Tema og undertema
        9. Er dette en specialregel eller undtagelse? (true/false)
        10. Kompleksitetsgrad (simpel, moderat, kompleks)
        
        Returner JSON:
        {{
          "chunks": [
            {{
              "content": "NØJAGTIG tekst fra kilden uden ændringer",
              "metadata": {{
                "doc_id": "{doc_id}",
                "type": "lovtekst/note/vejledning/osv",
                "paragraph": "§ X",
                "stykke": "Stk. Y",
                "concepts": ["nøgleord1", "nøgleord2", "nøgleord3"],
                "law_references": ["Ligningslovens § 33 A, stk. 1"],
                "case_references": ["SKM.2019.123"],
                "normalized_references": ["SKM.2019.123"],
                "theme": "tema",
                "subtheme": "undertema",
                "is_note": false,
                "is_exception": false,
                "affected_groups": ["skatteydere", "virksomheder", "rådgivere"],
                "complexity": "simpel/moderat/kompleks"
              }}
            }}
          ]
        }}
        """