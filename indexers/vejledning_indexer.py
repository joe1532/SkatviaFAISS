# indexers/vejledning_indexer.py
import re
import streamlit as st
import json
from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils, indexing
from utils.optimization import cached_call_gpt4o, process_segments_parallel

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Vejlednings-indekserer"
        self.description = "Specialiseret indeksering af skatteretlige vejledninger"
        
        self.vejledning_types = {
            "den_juridiske_vejledning": {
                "display_name": "Den Juridiske Vejledning",
                "template_name": "jv_template",
                "section_pattern": r'[A-Z]\.\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
            "styresignal": {
                "display_name": "Styresignal",
                "template_name": "styresignal_template",
                "section_pattern": r'\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            },
            "andet": {
                "display_name": "Anden vejledning",
                "template_name": "generisk_vejledning_template",
                "section_pattern": r'\d+(\.\d+)*',
                "example_pattern": r'Eksempel:',
                "case_reference_patterns": ["SKM", "TfS", "U", "LSRM"]
            }
        }
    
    def display_settings(self, st):
        """Viser indstillinger specifikt for vejledninger"""
        vejledning_type = st.selectbox(
            "Vejledningstype:",
            list(self.vejledning_types.keys()),
            format_func=lambda x: self.vejledning_types[x]["display_name"]
        )
        
        # Vejledningsspecifikke indstillinger
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
        
        return vejledning_type
    
    def process_document(self, text, doc_id, options):
        """Processer vejledningsdokument"""
        # Implementering specielt til vejledningsstruktur
        # Dette er en skitse der skal tilpasses
        
        # 1. Preprocessering - tilpasset til vejledninger
        processed_text = self._preprocess_vejledning(text)
        
        # 2. Segmentering - specialiseret til vejledningsstruktur
        segments, preserved_content, segment_stats = self._segment_vejledning(
            processed_text, max_segment_length=options.get("max_text_length", 30000)
        )
        
        # 3. Kontekstanalyse med vejledningsfokus
        with st.spinner("Analyserer vejledningens struktur og indhold..."):
            vejledning_type = options.get("doc_type_key", "andet")
            context_prompt = self.get_context_prompt_template(vejledning_type)
            context_prompt_with_text = context_prompt + "\n\nVejledning:\n" + segments[0]
            
            context_summary = cached_call_gpt4o(
                context_prompt_with_text,
                model=options.get("model", "gpt-4o")
            )
            if not context_summary:
                st.error("Kunne ikke generere vejledningsanalyse. Prøv igen.")
                return None, None
        
        # 4. Chunking med vejledningsspecifikke prompts
        with st.spinner("Opdeler vejledningen i meningsfulde chunks..."):
            chunks = process_segments_parallel(
                segments, 
                vejledning_type, 
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
    
    def _preprocess_vejledning(self, text):
        """Forbehandling specifikt for vejledninger"""
        # Fjern sidenumre og andre forstyrrende elementer
        text = re.sub(r'Side \d+ af \d+', '', text)
        
        # Normalisér afsnits- og punktnummerering 
        text = re.sub(r'([A-Z])\.(\d+)\.(\d+)', r'\1.\2.\3', text)
        
        # Standardisér eksempelformater
        text = re.sub(r'Eks(?:empel)?[:,.]', 'Eksempel:', text, flags=re.IGNORECASE)
        
        return text
    
    def _segment_vejledning(self, text, max_segment_length=30000):
        """Segmentering tilpasset vejledningsstruktur"""
        segments = []
        preserved_content = {"sections": {}, "examples": {}}
        
        # Del ved hovedafsnit (f.eks. A.1, C.2 osv.)
        main_sections = re.split(r'([A-Z]\.\d+(?:\s+[^A-Z\d]+))', text)
        
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
        example_pattern = r'(Eksempel:(?:.*?)(?=\n\n|\n[A-Z]\.\d+|\Z))'
        for segment in ' '.join(segments):
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
    
    def get_context_prompt_template(self, vejledning_type):
        """Bygger en kontekst-prompt skabelon specifikt til vejledninger"""
        if vejledning_type == "den_juridiske_vejledning":
            return """
            Du er en ekspert i dansk skatteret, særligt Den Juridiske Vejledning. 
            Læs hele teksten og opbyg en forståelse af vejledningens struktur, temaer og relationer.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Vejledningens hovedafsnit og underafsnit (identificeret ved nummerering som A.1.2)
            - Centrale temaer der behandles i vejledningen
            - Lovhenvisninger og referencer til paragraf/stykker i skattelovgivningen
            - Referencer til afgørelser og domme (SKM, TfS, mv.)
            - Eksempler og deres formål
            - Fortolkningsbidrag til lovgivningen
            - Administrativ praksis beskrevet i vejledningen
            
            Format:
            {
              "document_id": "unik_id_for_dokumentet",
              "document_type": "juridisk_vejledning", 
              "version_date": "YYYY-MM-DD",
              "summary": {
                "main_sections": ["A.1", "A.2"],
                "section_hierarchy": {"A.1": ["A.1.1", "A.1.2"]},
                "section_titles": {"A.1": "Titel for sektion A.1"},
                "key_concepts": ["nøgleord1", "nøgleord2"],
                "law_references": {
                  "§ 33 A": ["A.1.1", "A.2.3"],
                  "Kildeskattelovens § 1": ["A.3.2"]
                },
                "examples": {
                  "1": {
                    "location": "A.1.2",
                    "describes": "Eksempel på grænsegænger-situation",
                    "law_reference": "§ 33 A, stk. 1"
                  }
                },
                "case_references": {
                  "SKM.2019.123": {
                    "location": "A.2.1",
                    "relevance": "Etablerer praksis for..."
                  }
                },
                "administrative_practice": [
                  {
                    "theme": "Grænsegængere",
                    "sections": ["A.1.3"],
                    "description": "Administration af reglerne for grænsegængere"
                  }
                ]
              }
            }
            """
        else:
            # Default vejlednings-prompt for andre typer
            return """
            Du er en ekspert i dansk skatteret. Læs denne vejledning og opbyg en forståelse af dens struktur og indhold.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Vejledningens hovedpunkter og underpunkter
            - Centrale temaer der behandles
            - Lovhenvisninger og referencer
            - Eksempler og deres formål
            
            Format:
            {
              "document_id": "unik_id_for_dokumentet",
              "document_type": "vejledning", 
              "version_date": "YYYY-MM-DD",
              "summary": {
                "main_sections": ["1", "2", "3"],
                "section_hierarchy": {"1": ["1.1", "1.2"]},
                "key_concepts": ["nøgleord1", "nøgleord2"],
                "law_references": {"§ 33 A": ["sektion 1.1"]},
                "examples": {"1": {"location": "sektion 1.2", "describes": "Eksempel på..."}}
              }
            }
            """
    
    def get_indexing_prompt_template(self, vejledning_type, context_summary, doc_id, section_number):
        """Bygger en indekseringsprompt for vejledninger"""
        return f"""
        Du er en ekspert i dansk skatteret der skal indeksere skatteretlige vejledninger. 
        Du har fået denne kontekst: {json.dumps(context_summary, ensure_ascii=False)}.
        
        Dette er sektion {section_number} af dokumentet. Din opgave er at opdele denne sektion i semantisk meningsfulde chunks.
        
        Opdel teksten i semantisk meningsfulde chunks baseret på vejledningens struktur (punkter og underpunkter).
        CHUNK ALDRIG MIDT I EN SÆTNING. 
        
        VIGTIGSTE REGEL: Bevar den KOMPLETTE, UÆNDREDE tekst fra kilden. Lav ALDRIG opsummeringer eller parafraseringer.
        
        RETURNER DIN SVAR SOM JSON.
        
        Tildel følgende metadata til hvert chunk:
        1. Dokument-ID: "{doc_id}"
        2. Afsnitsnummer (f.eks. "A.1.2" eller "3.4")
        3. Afsnitstitel hvis den findes
        4. Nøgleord (maks 5 pr. chunk)
        5. Lovhenvisninger (hvilke paragraffer og stykker fortolkes)
        6. Er dette et eksempel? (true/false)
        7. Referencer til domme/afgørelser
        8. Normaliserede referencer (fx "SKM2006635SKAT" → "SKM.2006.635")
        9. Tema og undertema
        10. Chunk-type (indledning, definition, beskrivelse, eksempel, praksis)
        11. Administrativ praksis (beskrivelse af praksis hvis relevant)
        12. Målgruppe (hvem er vejledningen rettet mod)
        13. Kompleksitetsgrad (simpel, moderat, kompleks)
        
        Returner JSON:
        {{
          "chunks": [
            {{
              "content": "NØJAGTIG tekst fra kilden uden ændringer",
              "metadata": {{
                "doc_id": "{doc_id}",
                "section": "A.1.2",
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
        """Skaber relationer mellem vejledning og lovtekst"""
        # Dette ville kræve en database af indekserede lovparagraffer
        # Her vises en simplificeret version
        
        for chunk in chunks:
            if "law_references" in chunk["metadata"] and chunk["metadata"]["law_references"]:
                # Her ville man faktisk slå op i en database af lovtekst
                # For nu markerer vi bare relationen
                chunk["metadata"]["linked_to_law"] = True
        
        return chunks