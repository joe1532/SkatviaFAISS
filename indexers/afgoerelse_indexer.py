# indexers/afgoerelse_indexer.py
import re
import streamlit as st
import json
from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils, indexing
from utils.optimization import cached_call_gpt4o, process_segments_parallel

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Afgørelses-indekserer"
        self.description = "Specialiseret indeksering af domme og afgørelser"
        
        self.afgoerelse_types = {
            "skm": {
                "display_name": "SKM-afgørelse",
                "template_name": "skm_template",
                "case_reference_patterns": ["SKM"]
            },
            "tfs": {
                "display_name": "TfS-afgørelse",
                "template_name": "tfs_template",
                "case_reference_patterns": ["TfS"]
            },
            "lsr": {
                "display_name": "LSR-afgørelse",
                "template_name": "lsr_template",
                "case_reference_patterns": ["LSRM"]
            },
            "dom": {
                "display_name": "Domstolsafgørelse",
                "template_name": "dom_template",
                "case_reference_patterns": ["U"]
            }
        }
    
    def display_settings(self, st):
        """Viser indstillinger specifikt for afgørelser"""
        afgoerelse_type = st.selectbox(
            "Afgørelsestype:",
            list(self.afgoerelse_types.keys()),
            format_func=lambda x: self.afgoerelse_types[x]["display_name"]
        )
        
        st.session_state.extract_facts = st.checkbox(
            "Uddrag faktum separat", 
            value=True,
            help="Behandl sagens faktiske omstændigheder som separate chunks"
        )
        
        st.session_state.extract_judicial_reasoning = st.checkbox(
            "Uddrag begrundelse og resultat separat", 
            value=True,
            help="Behandl rettens begrundelse og resultat som separate chunks"
        )
        
        st.session_state.link_to_law = st.checkbox(
            "Opret links til omtalte love", 
            value=True,
            help="Find og link til relevante lovparagraffer"
        )
        
        return afgoerelse_type
    
    def process_document(self, text, doc_id, options):
        """Processer afgørelsesdokument"""
        # 1. Preprocessering - tilpasset til afgørelser
        processed_text = text
        
        # 2. Segmentering - specialiseret til afgørelsesstruktur
        segments, preserved_content, segment_stats = text_analysis.segment_text_for_processing(
            processed_text, max_segment_length=options.get("max_text_length", 30000)
        )
        st.session_state.preserved_content = preserved_content
        
        # 3. Kontekstanalyse med afgørelsesfokus
        with st.spinner("Analyserer afgørelsens struktur og indhold..."):
            afgoerelse_type = options.get("doc_type_key", "skm")
            context_prompt = self.get_context_prompt_template(afgoerelse_type)
            context_prompt_with_text = context_prompt + "\n\nAfgørelse:\n" + ' '.join(segments[:2])  # Kombiner første to segmenter for bedre kontekst
            
            context_summary = cached_call_gpt4o(
                context_prompt_with_text,
                model=options.get("model", "gpt-4o")
            )
            if not context_summary:
                st.error("Kunne ikke generere afgørelsesanalyse. Prøv igen.")
                return None, None
        
        # 4. Chunking med afgørelsesspecifikke prompts
        with st.spinner("Opdeler afgørelsen i meningsfulde chunks..."):
            chunks = process_segments_parallel(
                segments, 
                afgoerelse_type, 
                context_summary, 
                doc_id, 
                options,
                get_template_func=self.get_indexing_prompt_template
            )
        
        # 5. Normaliser referencer til love og andre afgørelser
        chunks = text_analysis.normalize_case_references(chunks)
        
        return chunks, context_summary
    
    def get_context_prompt_template(self, afgoerelse_type):
        """Bygger en kontekst-prompt skabelon specifikt til afgørelser"""
        if afgoerelse_type == "skm":
            return """
            Du er en ekspert i dansk skatteret, særligt i analyse af skattemæssige afgørelser. 
            Læs hele afgørelsen og opbyg en forståelse af dens struktur, temaer og juridiske principper.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Afgørelsens identifikation (SKM-nummer)
            - Afgørelsesdato
            - Afgørelsestype (bindende svar, administrativ afgørelse, domstolsafgørelse)
            - Retsinstans (SKAT, Landsskatteretten, Landsret, Højesteret)
            - Hovedtemaer i afgørelsen
            - Centrale problemstillinger
            - Relevante lovbestemmelser (hvilke paragraffer fortolkes)
            - Referencer til andre afgørelser
            - Procesforløb (inkl. sagsforløb ved tidligere instanser)
            - Parternes påstande
            - Sagens faktiske omstændigheder
            - Juridisk analyse og begrundelse
            - Resultat og konklusion
            - Præcedensskabende betydning
            
            Format:
            {
              "document_id": "skm_nummer",
              "document_type": "afgørelse", 
              "version_date": "YYYY-MM-DD",
              "deciding_authority": "Landsskatteretten/Østre Landsret/osv.",
              "summary": {
                "main_themes": ["tema1", "tema2"],
                "legal_issues": ["problemstilling1", "problemstilling2"],
                "law_references": ["Ligningslovens § 33 A", "Kildeskattelovens § 1"],
                "case_references": ["SKM.2018.123", "TfS.2019.456"],
                "normalized_references": ["SKM.2018.123", "TfS.2019.456"],
                "facts_summary": "Kort beskrivelse af faktiske omstændigheder",
                "legal_reasoning": "Kort beskrivelse af juridisk begrundelse",
                "conclusion": "Kort beskrivelse af resultat",
                "precedent_value": "høj/medium/lav",
                "document_structure": {
                  "section1": "Sagsfremstilling",
                  "section2": "Parternes påstande",
                  "section3": "Begrundelse og resultat"
                }
              }
            }
            """
        else:
            # Default afgørelses-prompt
            return """
            Du er en ekspert i dansk skatteret. Læs denne afgørelse og opbyg en forståelse af dens struktur og juridiske principper.
            
            RETURNER DIN SVAR SOM JSON.
            
            Returner en JSON-opsummering med:
            - Afgørelsens identifikation (nummer/reference)
            - Afgørelsesdato
            - Afgørelsestype
            - Retsinstans
            - Hovedtemaer
            - Relevante lovbestemmelser
            - Sagens faktiske omstændigheder
            - Begrundelse og resultat
            
            Format:
            {
              "document_id": "afgørelses_id",
              "document_type": "afgørelse", 
              "version_date": "YYYY-MM-DD",
              "deciding_authority": "instans",
              "summary": {
                "main_themes": ["tema1", "tema2"],
                "law_references": ["Ligningslovens § 33 A"],
                "case_references": ["SKM.2019.123"],
                "facts_summary": "Kort beskrivelse",
                "conclusion": "Resultat"
              }
            }
            """
    
    def get_indexing_prompt_template(self, afgoerelse_type, context_summary, doc_id, section_number):
        """Bygger en indekseringsprompt for afgørelser"""
        return f"""
        Du er en ekspert i dansk skatteret der skal indeksere skatteretlige afgørelser. 
        Du har fået denne kontekst: {json.dumps(context_summary, ensure_ascii=False)}.
        
        Dette er sektion {section_number} af afgørelsen. Din opgave er at opdele denne sektion i semantisk meningsfulde chunks.
        
        Opdel teksten i semantisk meningsfulde chunks baseret på afgørelsens struktur.
        CHUNK ALDRIG MIDT I EN SÆTNING. 
        
        VIGTIGSTE REGEL: Bevar den KOMPLETTE, UÆNDREDE tekst fra kilden. Lav ALDRIG opsummeringer eller parafraseringer.
        
        RETURNER DIN SVAR SOM JSON.
        
        Tildel følgende metadata til hvert chunk:
        1. Dokument-ID: "{doc_id}"
        2. Sektion (sagsfremstilling, parternes påstande, begrundelse, resultat)
        3. Nøglekoncepter (maks 5 pr. chunk)
        4. Lovhenvisninger (hvilke paragraffer fortolkes)
        5. Referencer til andre afgørelser
        6. Normaliserede referencer
        7. Tema og undertema
        8. Chunk-type (faktum, påstand, begrundelse, resultat)
        9. Juridiske principper der anvendes
        10. Juridiske argumenter
        11. Kompleksitetsgrad (simpel, moderat, kompleks)
        12. Præcedensskabende betydning
        
        Returner JSON:
        {{
          "chunks": [
            {{
              "content": "NØJAGTIG tekst fra kilden uden ændringer",
              "metadata": {{
                "doc_id": "{doc_id}",
                "section": "sagsfremstilling/påstand/begrundelse/resultat",
                "concepts": ["nøgleord1", "nøgleord2", "nøgleord3"],
                "law_references": ["Ligningslovens § 33 A, stk. 1"],
                "case_references": ["SKM.2019.123", "TfS.2018.456"],
                "normalized_references": ["SKM.2019.123", "TfS.2018.456"],
                "theme": "tema",
                "subtheme": "undertema",
                "chunk_type": "faktum/påstand/begrundelse/resultat",
                "legal_principles": ["princip1", "princip2"],
                "legal_arguments": ["argument1", "argument2"],
                "complexity": "simpel/moderat/kompleks",
                "precedent_value": "høj/medium/lav"
              }}
            }}
          ]
        }}
        """