# indexers/cirkulaere_indexer.py
import re
import streamlit as st
import json
from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils, indexing
from utils.optimization import cached_call_gpt4o, process_segments_parallel

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Cirkulære-indekserer"
        self.description = "Specialiseret indeksering af cirkulærer"
        
        self.cirkulaere_types = {
            "skatte_cirkulaere": {
                "display_name": "Skattecirkulære",
                "template_name": "skatte_cirkulaere_template",
                "section_pattern": r'\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
            "told_cirkulaere": {
                "display_name": "Toldcirkulære",
                "template_name": "told_cirkulaere_template",
                "section_pattern": r'\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
            "andet_cirkulaere": {
                "display_name": "Andet cirkulære",
                "template_name": "andet_cirkulaere_template",
                "section_pattern": r'\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            }
        }
    
    def display_settings(self, st):
        """Viser indstillinger specifikt for cirkulærer"""
        cirkulaere_type = st.selectbox(
            "Cirkulæretype:",
            list(self.cirkulaere_types.keys()),
            format_func=lambda x: self.cirkulaere_types[x]["display_name"]
        )
        
        # Cirkulærespecifikke indstillinger
        st.session_state.extract_examples = st.checkbox(
            "Indekser eksempler separat", 
            value=True,
            help="Behandl eksempler som separate chunks med ekstra metadata"
        )
        
        st.session_state.link_to_law = st.checkbox(
            "Opret links til lovtekst", 
            value=True,
            help="Find og link til relaterede lovparagraffer"
        )
        
        st.session_state.cluster_by_sections = st.checkbox(
            "Gruppér efter afsnit", 
            value=True,
            help="Opdel cirkulæret efter dets numeriske afsnitsstruktur"
        )
        
        return cirkulaere_type
    
    def process_document(self, text, doc_id, options):
        """Processer cirkulæredokument"""
        # 1. Preprocessering - tilpasset til cirkulærer
        processed_text = self._preprocess_cirkulaere(text)
        
        # 2. Segmentering - specialiseret til cirkulærestruktur
        segments, preserved_content, segment_stats = self._segment_cirkulaere(
            processed_text, max_segment_length=options.get("max_text_length", 30000)
        )
        st.session_state.preserved_content = preserved_content
        
        # 3. Kontekstanalyse med cirkulærefokus
        with st.spinner("Analyserer cirkulærets struktur og indhold..."):
            cirkulaere_type = options.get("doc_type_key", "skatte_cirkulaere")
            context_prompt = self.get_context_prompt_template(cirkulaere_type)
            context_prompt_with_text = context_prompt + "\n\nCirkulære:\n" + segments[0]
            
            context_summary = cached_call_gpt4o(
                context_prompt_with_text,
                model=options.get("model", "gpt-4o")
            )
            if not context_summary:
                st.error("Kunne ikke generere cirkulæreanalyse. Prøv igen.")
                return None, None
        
        # 4. Chunking med cirkulærespecifikke prompts
        with st.spinner("Opdeler cirkulæret i meningsfulde chunks..."):
            chunks = process_segments_parallel(
                segments, 
                cirkulaere_type, 
                context_summary, 
                doc_id, 
                options,
                get_template_func=self.get_indexing_prompt_template
            )
        
        # 5. Identifikation af eksempler og referencer
        with st.spinner("Identificerer eksempler og juridiske referencer..."):
            chunks = self._extract_examples_and_references(chunks)
            
            # Find lovhenvisninger og skab relationer
            if st.session_state.link_to_law:
                chunks = self._link_to_law_paragraphs(chunks)
        
        return chunks, context_summary
    
    def _preprocess_cirkulaere(self, text):
        """Forbehandling specifikt for cirkulærer"""
        # Fjern sidenumre og andre forstyrrende elementer
        text = re.sub(r'Side \d+ af \d+', '', text)
        
        # Normalisér afsnits- og punktnummerering 
        text = re.sub(r'(\d+)\.(\d+)(\s+[A-Za-z])', r'\1.\2.\3', text)
        
        # Standardisér eksempelformater
        text = re.sub(r'Eks(?:empel)?[:,.]', 'Eksempel:', text, flags=re.IGNORECASE)
        
        return text
    
    def _segment_cirkulaere(self, text, max_segment_length=30000):
        """Segmentering tilpasset cirkulærestruktur"""
        segments = []
        preserved_content = {"sections": {}, "examples": {}}
        
        # Del ved hovedafsnit (f.eks. 1., 2. osv.)
        main_sections = re.split(r'(\d+\.\s+[^0-9]+)', text)
        
        current_segment = ""
        for i in range(0, len(main_sections)-1, 2):
            if i+1 < len(main_sections):
                section_header = main_sections[i]
                section_content = main_sections[i+1]
                full_section = section_header + section_content
                
                # Bevar original sektions-tekst
                section_id = section_header.strip()
                preserved_content["sections"][section_id] = full_section
                
                # Del i passende segmenter
                if len(current_segment + full_section) > max_segment_length:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = full_section
                else:
                    current_segment += full_section
        
        # Tilføj sidste segment
        if current_segment:
            segments.append(current_segment)
        
        # Udpak eksempler
        example_pattern = r'(Eksempel:(?:.*?)(?=\n\n|\n\d+\.|\Z))'
        for segment in segments:
            for match in re.finditer(example_pattern, segment, re.DOTALL):
                example_text = match.group(1)
                example_id = f"eks_{len(preserved_content['examples'])+1}"
                preserved_content["examples"][example_id] = example_text
        
        stats = {
            "segments": len(segments),
            "preserved_sections": len(preserved_content["sections"]),
            "preserved_examples": len(preserved_content["examples"])
        }
        
        return segments, preserved_content, stats
    
    def get_context_prompt_template(self, cirkulaere_type):
        """Bygger en kontekst-prompt skabelon specifikt til cirkulærer"""
        if cirkulaere_type == "skatte_cirkulaere":
            return """
            Du er en ekspert i dansk skatteret, særligt cirkulærer. 
            Læs hele teksten og opbyg en forståelse af cirkulærets struktur, temaer og relationer.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Cirkulærets nummer og dato
            - Hovedafsnit og underafsnit (identificeret ved nummerering som 1.2.3)
            - Centrale temaer der behandles i cirkulæret
            - Lovhenvisninger og referencer til paragraffer i skattelovgivningen
            - Eksempler og deres formål
            - Administrativ praksis beskrevet i cirkulæret
            
            Format:
            {
              "document_id": "cirkulære_nummer",
              "document_type": "cirkulære", 
              "version_date": "YYYY-MM-DD",
              "summary": {
                "main_sections": ["1", "2", "3"],
                "section_hierarchy": {"1": ["1.1", "1.2"]},
                "section_titles": {"1": "Titel for afsnit 1"},
                "key_concepts": ["nøgleord1", "nøgleord2"],
                "law_references": {
                  "§ 33 A": ["afsnit 1.1", "afsnit 2.3"],
                  "Kildeskattelovens § 1": ["afsnit 3.2"]
                },
                "examples": {
                  "1": {
                    "location": "afsnit 1.2",
                    "describes": "Eksempel på...",
                    "law_reference": "§ X, stk. Y"
                  }
                },
                "administrative_practice": [
                  {
                    "theme": "Tema",
                    "sections": ["1.3"],
                    "description": "Beskrivelse af praksis"
                  }
                ]
              }
            }
            """
        else:
            # Default cirkulære-prompt
            return """
            Du er en ekspert i dansk skatteret. Læs dette cirkulære og opbyg en forståelse af dets struktur og indhold.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Cirkulærets nummer og dato
            - Hovedpunkter og underpunkter
            - Centrale temaer der behandles
            - Lovhenvisninger og referencer
            - Eksempler og deres formål
            
            Format:
            {
              "document_id": "cirkulære_nummer",
              "document_type": "cirkulære", 
              "version_date": "YYYY-MM-DD",
              "summary": {
                "main_sections": ["1", "2", "3"],
                "section_hierarchy": {"1": ["1.1", "1.2"]},
                "key_concepts": ["nøgleord1", "nøgleord2"],
                "law_references": {"§ 33 A": ["afsnit 1.1"]},
                "examples": {"1": {"location": "afsnit 1.2", "describes": "Eksempel på..."}}
              }
            }
            """
    
    def get_indexing_prompt_template(self, cirkulaere_type, context_summary, doc_id, section_number):
        """Bygger en indekseringsprompt for cirkulærer"""
        return f"""
        Du er en ekspert i dansk skatteret der skal indeksere skatteretlige cirkulærer. 
        Du har fået denne kontekst: {json.dumps(context_summary, ensure_ascii=False)}.
        
        Dette er sektion {section_number} af cirkulæret. Din opgave er at opdele denne sektion i semantisk meningsfulde chunks.
        
        Opdel teksten i semantisk meningsfulde chunks baseret på cirkulærets struktur (punkter og underpunkter).
        CHUNK ALDRIG MIDT I EN SÆTNING. 
        
        VIGTIGSTE REGEL: Bevar den KOMPLETTE, UÆNDREDE tekst fra kilden. Lav ALDRIG opsummeringer eller parafraseringer.
        
        RETURNER DIN SVAR SOM JSON.
        
        Tildel følgende metadata til hvert chunk:
        1. Dokument-ID: "{doc_id}"
        2. Afsnitsnummer (f.eks. "1.2.3")
        3. Afsnitstitel hvis den findes
        4. Nøgleord (maks 5 pr. chunk)
        5. Lovhenvisninger (hvilke paragraffer omtales)
        6. Er dette et eksempel? (true/false)
        7. Referencer til domme/afgørelser
        8. Normaliserede referencer
        9. Tema og undertema
        10. Chunk-type (indledning, definition, beskrivelse, eksempel, praksis)
        11. Administrativ praksis (beskrivelse af praksis hvis relevant)
        12. Målgruppe (hvem er cirkulæret rettet mod)
        13. Kompleksitetsgrad (simpel, moderat, kompleks)
        
        Returner JSON:
        {{
          "chunks": [
            {{
              "content": "NØJAGTIG tekst fra kilden uden ændringer",
              "metadata": {{
                "doc_id": "{doc_id}",
                "section": "1.2.3",
                "section_title": "Titel på afsnittet",
                "concepts": ["nøgleord1", "nøgleord2", "nøgleord3", "nøgleord4", "nøgleord5"],
                "law_references": ["Ligningslovens § 33 A, stk. 1", "Kildeskattelovens § 1"],
                "is_example": false,
                "case_references": ["SKM.2019.123", "TfS.2018.456"],
                "normalized_references": ["SKM.2019.123", "TfS.2018.456"],
                "theme": "tema",
                "subtheme": "undertema",
                "chunk_type": "beskrivelse/eksempel/praksis",
                "administrative_practice": "beskrivelse af praksis eller null",
                "target_audience": ["skatteydere", "virksomheder", "rådgivere"],
                "complexity": "simpel/moderat/kompleks"
              }}
            }}
          ]
        }}
        """
    
    def _extract_examples_and_references(self, chunks):
        """Udtrækker eksempler og lovhenvisninger fra chunks"""
        updated_chunks = []
        
        for chunk in chunks:
            updated_chunk = chunk.copy()
            content = chunk.get("content", "")
            
            # Identificer eksempler
            if re.search(r'Eksempel:', content, re.IGNORECASE):
                updated_chunk["metadata"]["is_example"] = True
                updated_chunk["metadata"]["chunk_type"] = "eksempel"
            
            # Find lovhenvisninger
            law_refs = []
            patterns = [
                r'((?:lignings|person|kilde|selskabs)lovens?)\s+§\s*(\d+\s*[A-Za-z]?)(?:,?\s*(?:stk\.|stykke)\s*(\d+))?',
                r'(§\s*\d+\s*[A-Za-z]?)(?:,?\s*(?:stk\.|stykke)\s*(\d+))?'
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    if len(match.groups()) >= 2 and match.group(2):
                        if match.group(1).lower().startswith('§'):
                            # Direkte paragrafhenvisning
                            ref = match.group(1)
                            if len(match.groups()) >= 3 and match.group(3):
                                ref += f", stk. {match.group(3)}"
                            law_refs.append(ref)
                        else:
                            # Lov + paragraf
                            lov = match.group(1)
                            para = match.group(2)
                            ref = f"{lov} § {para}"
                            if len(match.groups()) >= 3 and match.group(3):
                                ref += f", stk. {match.group(3)}"
                            law_refs.append(ref)
            
            if law_refs:
                updated_chunk["metadata"]["law_references"] = law_refs
            
            updated_chunks.append(updated_chunk)
        
        return updated_chunks
    
    def _link_to_law_paragraphs(self, chunks):
        """Skaber relationer mellem cirkulære og lovtekst"""
        # Dette ville kræve en database af indekserede lovparagraffer
        # Her vises en simplificeret version
        
        for chunk in chunks:
            if "law_references" in chunk["metadata"] and chunk["metadata"]["law_references"]:
                # Her ville man faktisk slå op i en database af lovtekst
                # For nu markerer vi bare relationen
                chunk["metadata"]["linked_to_law"] = True
        
        return chunks