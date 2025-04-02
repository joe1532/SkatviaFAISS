# indexers/juridisk_vejledning_indexer.py
import re
import streamlit as st
import json
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

from .base_indexer import BaseIndexer
from utils import api_utils, text_analysis, validation, pdf_utils
from utils.optimization import cached_call_gpt4o, process_segments_parallel

class Indexer(BaseIndexer):
    def __init__(self):
        super().__init__()
        self.name = "Juridisk Vejledning Indekserer"
        self.description = "Generel indeksering af juridiske vejledninger"
        
        # Dynamiske konfigurationsværdier (erstattes efter domæneanalyse)
        self.domain_config = {}
        self.question_patterns = {}  
        self.law_abbreviations = {}
        self.person_groups = {}
        self.default_themes = ["juridisk vejledning"]
        
        # Opsæt logging
        self.logger = logging.getLogger("juridisk_vejledning_indexer")
    
    def display_settings(self, st):
        """Viser indstillinger for juridisk vejledning indekseringen"""
        st.write("### Indstillinger for juridisk vejledning")
        
        st.session_state.extract_examples = st.checkbox(
            "Indekser eksempler separat", 
            value=True,
            help="Behandl eksempler som separate chunks med ekstra metadata"
        )
        
        st.session_state.extract_case_tables = st.checkbox(
            "Indekser domsoversigter separat", 
            value=True,
            help="Behandl tabeller med domme og afgørelser som separate chunks"
        )
        
        st.session_state.extract_subsections = st.checkbox(
            "Indekser efter underafsnit", 
            value=True,
            help="Del hvert hovedafsnit i underafsnit (f.eks. 'Regel', 'Dokumentation')"
        )
        
        st.session_state.balance_chunks = st.checkbox(
            "Balancér chunks", 
            value=True,
            help="Optimér chunk-størrelse for bedre søgning"
        )
        
        st.session_state.semantic_chunking = st.checkbox(
            "Semantisk chunking", 
            value=True,
            help="Anvend forbedret semantisk chunking der bevarer juridiske ræsonnementer"
        )
        
        # Indstillinger for chunk-størrelser
        st.session_state.min_chunk_size = st.slider(
            "Minimum chunk-størrelse (tegn)",
            min_value=100,
            max_value=500,
            value=250,
            step=50,
            help="Chunks mindre end dette vil blive forsøgt slået sammen"
        )
        
        st.session_state.target_chunk_size = st.slider(
            "Optimal chunk-størrelse (tegn)",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Tilstræbt størrelse på chunks for optimal søgning"
        )
        
        # Valgfrit: Angiv version af Den Juridiske Vejledning
        jv_version = st.text_input(
            "Version af juridisk vejledning (ÅÅÅÅ-MM-DD)",
            value="",
            help="Angiv versionsdatoen for vejledningen, f.eks. 2022-08-01"
        )
        
        # Gem versionen i session state
        st.session_state.jv_version = jv_version
        
        return "juridisk_vejledning"
    
    def process_document(self, text, doc_id, options):
        """
        Processer juridisk vejledning med dynamisk domænetilpasset indeksering
        """
        # Opdater statistik
        processing_stats = {}
        
        try:
            # Fase 1: Dynamisk domæneanalyse
            with st.spinner("Analyserer dokumentets juridiske område..."):
                self.domain_config = self._analyze_domain(text[:20000], options.get("model", "gpt-4o"))
                # Opdater indekserens konfiguration med den dynamiske konfiguration
                self._update_indexer_config(self.domain_config)
                st.write(f"Juridisk område identificeret med {len(self.question_patterns)} spørgsmålstyper og {len(self.law_abbreviations)} lovforkortelser")
            
            # 1. Preprocessering - generisk rensning for juridiske tekster
            processed_text = self._preprocess_text(text)
            
            # 2. Segmentering efter hovedafsnit (C.X.X.X)
            segments, preserved_content = self._segment_by_sections(processed_text)
            st.session_state.preserved_content = preserved_content
            
            processing_stats["sections_count"] = len(segments)
            processing_stats["total_length"] = len(processed_text)
            
            # 3. Strukturanalyse med AI
            with st.spinner("Analyserer dokumentets struktur og indhold..."):
                context_summary = self._analyze_structure(
                    segments[0] if segments else processed_text[:10000],
                    options.get("model", "gpt-4o")
                )
                
                # Tilføj domæneanalyse til context summary
                context_summary["domain_config"] = self.domain_config
                
                # Hvis vi kan finde versionsdato i dokumentet, brug den
                version_date = self._extract_version_date(processed_text)
                if version_date:
                    context_summary["version_date"] = version_date
                elif hasattr(st.session_state, 'jv_version') and st.session_state.jv_version:
                    context_summary["version_date"] = st.session_state.jv_version
                    
                context_summary["document_type"] = "juridisk_vejledning"
                context_summary["doc_id"] = doc_id
            
            # 4. Processering af alle segmenter
            with st.spinner(f"Analyserer {len(segments)} afsnit fra dokumentet..."):
                all_chunks = []
                
                # Vis en progressbar
                progress_bar = st.progress(0)
                
                for i, segment in enumerate(segments):
                    try:
                        # Processer dette segment
                        section_id = self._extract_section_id(segment)
                        
                        # Vis fremskridt
                        st.write(f"Processerer afsnit {section_id if section_id else f'{i+1}/{len(segments)}'}")
                        
                        # Processer segmentet til chunks
                        segment_chunks = self._process_segment(
                            segment, 
                            context_summary, 
                            doc_id, 
                            section_id, 
                            self._extract_section_title(segment, section_id), 
                            options
                        )
                        
                        all_chunks.extend(segment_chunks)
                    except Exception as e:
                        self.logger.error(f"Fejl ved processering af segment {i}: {str(e)}")
                        st.error(f"Advarsel: Problem med afsnit {i+1}. Fortsætter med næste afsnit.")
                    
                    # Opdater progressbar
                    progress_bar.progress((i + 1) / len(segments))
            
            # 5. Balancér chunklængder hvis aktiveret
            if hasattr(st.session_state, 'balance_chunks') and st.session_state.balance_chunks:
                with st.spinner("Balancerer chunks for optimal søgning..."):
                    all_chunks = self._balance_chunks(all_chunks)
            
            # 6. Efterbehandling af chunks
            with st.spinner("Efterbehandler chunks..."):
                # Tilføj krydsreferencer mellem chunks
                all_chunks = self._add_cross_references(all_chunks)
                
                # Normaliser lovhenvisninger
                all_chunks = self._normalize_law_references(all_chunks)
                
                # Normaliser domsreferencer
                all_chunks = self._normalize_case_references(all_chunks)
                
                # Reparer manglende felter
                all_chunks = self._ensure_complete_metadata(all_chunks)
                
                # Tilføj juridisk status og fortolkning
                all_chunks = self._add_legal_status(all_chunks)
                
                # Tilføj information om chunks til statistik
                processing_stats["chunks_count"] = len(all_chunks)
                processing_stats["example_chunks"] = len([c for c in all_chunks if c["metadata"].get("is_example", False)])
                processing_stats["law_chunks"] = len([c for c in all_chunks if c["metadata"].get("law_references", [])])
                processing_stats["case_chunks"] = len([c for c in all_chunks if c["metadata"].get("case_references", [])])
                
                # Beregn gennemsnitlig chunk-størrelse og standardafvigelse
                chunk_sizes = [len(c["content"]) for c in all_chunks]
                processing_stats["avg_chunk_size"] = np.mean(chunk_sizes) if chunk_sizes else 0
                processing_stats["std_chunk_size"] = np.std(chunk_sizes) if chunk_sizes else 0
                processing_stats["min_chunk_size"] = min(chunk_sizes) if chunk_sizes else 0
                processing_stats["max_chunk_size"] = max(chunk_sizes) if chunk_sizes else 0
            
            # Opdater context_summary med processing_stats
            context_summary["processing_stats"] = processing_stats
            
            return all_chunks, context_summary
            
        except Exception as e:
            self.logger.error(f"Fejl under dokumentprocessering: {str(e)}")
            st.error(f"Der opstod en fejl under indekseringen: {str(e)}")
            # Returner tomme resultater ved fejl
            return [], {"error": str(e), "doc_id": doc_id}

    def _analyze_domain(self, text_sample, model="gpt-4o"):
        """Analyserer retsområdet for at identificere domænespecifikke elementer"""
        prompt = """
        Du er ekspert i dansk jura. Analyser denne del af en juridisk vejledning og identificér:
        
        1. Hvilket juridisk område teksten handler om
        2. Lovforkortelser der anvendes (fx "LL" for ligningsloven)
        3. Persongrupper eller entiteter der omtales i dette retsområde
        4. Typer af faglige spørgsmål der er relevante for dette retsområde
        5. Specialregler eller undtagelser specifikke for dette område
        6. Centrale temaer for retsområdet
        
        RETURNER DIN ANALYSE SOM JSON med følgende struktur:
        {
          "legal_domain": "Retsområdets navn",
          "law_abbreviations": {
             "lovnavn1": "forkortelse1",
             "lovnavn2": "forkortelse2"
          },
          "person_groups": {
             "gruppe1": ["synonym1", "synonym2"],
             "gruppe2": ["synonym3", "synonym4"]
          },
          "question_patterns": {
             "spørgsmål_type1": ["mønster1", "mønster2"],
             "spørgsmål_type2": ["mønster3", "mønster4"]
          },
          "legal_exceptions": ["regel1", "regel2"],
          "standard_themes": ["tema1", "tema2"]
        }
        """
        
        try:
            # Kald sprogmodel
            domain_config = cached_call_gpt4o(
                prompt + "\n\nTekst:\n" + text_sample, 
                model=model
            )
            return domain_config
        except Exception as e:
            self.logger.error(f"Fejl ved domæneanalyse: {str(e)}")
            # Returner standardkonfiguration ved fejl
            return {
                "legal_domain": "Ukendt juridisk område",
                "law_abbreviations": {},
                "person_groups": {},
                "question_patterns": {
                    "general": ["hvornår", "hvordan", "hvilke regler"]
                },
                "legal_exceptions": [],
                "standard_themes": ["juridisk vejledning"]
            }

    def _update_indexer_config(self, domain_config):
        """Opdaterer indekserens konfiguration med domænespecifikke elementer"""
        if "law_abbreviations" in domain_config and domain_config["law_abbreviations"]:
            self.law_abbreviations = domain_config["law_abbreviations"]
        
        if "question_patterns" in domain_config and domain_config["question_patterns"]:
            self.question_patterns = domain_config["question_patterns"]
        
        if "person_groups" in domain_config and domain_config["person_groups"]:
            self.person_groups = domain_config["person_groups"]
        
        if "standard_themes" in domain_config and domain_config["standard_themes"]:
            self.default_themes = domain_config["standard_themes"]
            
        self.logger.info(f"Indekseringskonfiguration opdateret for domæne: {domain_config.get('legal_domain', 'Ukendt')}")

    def _preprocess_text(self, text):
        """Forbehandling generaliseret til juridiske vejledninger"""
        # Fjern sidefødder og -hoveder med robust mønstergenkendelse
        header_footer_patterns = [
            r'Printet fra (?:Karnov|SKAT).*?licensvilkårene',
            r'Side \d+ af \d+',
            r'(?:Opdateret: |Version: )\d{1,2}\.\d{1,2}\.\d{4}',
            r'Copyright © \d{4} (?:Karnov Group|SKAT)'
        ]
        
        for pattern in header_footer_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Standardiser afsnitsoverskrifter (generelt mønster for afsnit som C.A.X.X.X)
        text = re.sub(r'([A-Z]\.[A-Z]\.\d+\.\d+\.\d+)(\s+[A-Za-z])', r'\1\2', text)
        
        # Standardiser underafsnitsoverskrifter
        text = re.sub(r'\n([A-Za-z][\w\s]+)\n-+\n', r'\n\1\n', text)
        
        # Standardiser interne henvisninger
        text = re.sub(r'Se også\s*\n', r'Se også\n', text)
        text = re.sub(r'jf\.\s*', r'jf. ', text)
        text = re.sub(r'Bemærk\s*\n', r'Bemærk\n', text)
        
        # Standardiser eksempelformater
        text = re.sub(r'Eksempel\s*(\d+)[:.]', r'Eksempel \1:', text)
        
        # Standardiser paragrafformater
        text = re.sub(r'§\s*(\d+[a-zA-Z]?)', r'§ \1', text)
        text = re.sub(r'stk\.\s*(\d+)', r'stk. \1', text)
        
        # Fjern dobbelte linjeskift
        text = re.sub(r'\n\s*\n\s*\n', r'\n\n', text)
        
        # Fjern unødvendige mellemrum
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        return text

    def _segment_by_sections(self, text):
        """Opdeler teksten i segmenter baseret på hovedafsnit (generaliseret mønster)"""
        # Find alle hovedafsnit med generelt regex mønster der matcher juridiske afsnitsformater
        # Dette mønster er generaliseret til at fange både C.F.X.X.X, C.A.X.X.X osv.
        section_pattern = r'([A-Z]\.[A-Z]\.\d+\.\d+\.\d+\s+.+?)(?=[A-Z]\.[A-Z]\.\d+\.\d+\.\d+|$)'
        matches = list(re.finditer(section_pattern, text, re.DOTALL))
    
        # Hvis ingen afsnit blev fundet, returner hele teksten som ét segment
        if not matches:
            return [text], {"sections": {}}
    
        # Opdel teksten i segmenter
        segments = []
        preserved_content = {"sections": {}, "hierarchical_structure": {}}
    
        # Identificer hierarkiet i dokumentet
        section_hierarchy = {}
    
        for match in matches:
            segment = match.group(1)
            segments.append(segment)
        
            # Uddrag afsnits-ID
            section_id_match = re.search(r'([A-Z]\.[A-Z]\.\d+\.\d+\.\d+)', segment)
            if section_id_match:
                section_id = section_id_match.group(1)
                preserved_content["sections"][section_id] = segment
            
                # Opbyg hierarki
                parts = section_id.split('.')
                if len(parts) >= 4:  # A.B.1.2
                    parent_id = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    if parent_id not in section_hierarchy:
                        section_hierarchy[parent_id] = []
                    section_hierarchy[parent_id].append(section_id)
    
        preserved_content["hierarchical_structure"] = section_hierarchy
    
        return segments, preserved_content

    def _extract_section_id(self, segment):
        """Udtrækker afsnits-ID fra et segment (generaliseret til forskellige formater)"""
        # Generaliseret mønster for juridiske vejledninger
        match = re.search(r'([A-Z]\.[A-Z]\.\d+\.\d+\.\d+)', segment)
        if match:
            return match.group(1)
        return None

    def _extract_section_title(self, segment, section_id):
        """Udtrækker afsnitstitel fra et segment"""
        if not section_id:
            return None
            
        title_match = re.search(
            r'{}\s+([^\n]+)'.format(re.escape(section_id)), 
            segment
        )
        if title_match:
            return title_match.group(1).strip()
        return None

    def _extract_version_date(self, text):
        """Udtrækker versionsdato fra vejledningsteksten (generaliseret)"""
        # Søg efter forskellige datoangivelsesformater i starten af dokumentet
        date_patterns = [
            r'(?:Juridisk vejledning|Version)[\s:]+(\d{4}-\d{2}-\d{2})',
            r'(?:Version|Gældende fra)[\s:]+(\d{1,2}\.\d{1,2}\.\d{4})',
            r'(?:Opdateret|Udgivet)[\s:]+(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _analyze_structure(self, text, model="gpt-4o"):
        """Analyserer strukturen af vejledningen dynamisk med en sprogmodel"""
        prompt = """
        Du er en ekspert i dansk jura. Analyser denne del af den juridiske vejledning og opbyg en forståelse af dens struktur.
        
        RETURNER DIN ANALYSE SOM JSON.
        
        Lav en JSON-opsummering med:
        - Dokumentets hierarkiske struktur (afsnit og underafsnit)
        - Hovedtemaer i hvert afsnit
        - Lovhenvisninger (hvilke paragraffer og stykker fortolkes)
        - Referencer til domme og afgørelser
        - Centrale juridiske begreber der omtales
        - Persongrupper der omtales
        - Juridiske undtagelser og specialregler
        
        Format:
        {
          "structure": {
            "main_sections": ["A.B.1.2", "A.B.1.3"],
            "section_hierarchy": {
              "A.B.1": ["A.B.1.2", "A.B.1.3"]
            },
            "section_titles": {
              "A.B.1.2": "Titel på afsnit",
              "A.B.1.3": "Titel på andet afsnit"
            }
          },
          "themes": {
            "A.B.1.2": ["tema1", "tema2"]
          },
          "law_references": {
            "LOV § X, stk. Y": ["A.B.1.2"],
            "LOV § Z": ["A.B.1.2"]
          },
          "case_references": {
            "DOM2011.747": {
              "sections": ["A.B.1.2"],
              "summary": "Kort beskrivelse"
            }
          },
          "key_concepts": ["begreb1", "begreb2"],
          "affected_groups": ["gruppe1", "gruppe2"],
          "legal_exceptions": [
            {
              "rule": "Hovedregel beskrivelse",
              "exception": "Undtagelse beskrivelse",
              "sections": ["A.B.1.3"] 
            }
          ]
        }
        """
        
        # Tilføj teksten til prompten
        prompt_with_text = prompt + "\n\nJuridisk Vejledning (uddrag):\n" + text[:8000]
        
        try:
            # Kald sprogmodel med caching
            result = cached_call_gpt4o(prompt_with_text, model=model)
            
            # Tilføj title baseret på første afsnit
            title_match = re.search(r'[A-Z]\.[A-Z]\.\d+\.\d+\s+(.*?)(?:\n|$)', text[:1000])
            if title_match:
                title = title_match.group(1).strip()
                result["title"] = title
                
            return result
        except Exception as e:
            self.logger.error(f"Fejl ved strukturanalyse: {str(e)}")
            # Returner standardstruktur ved fejl
            return {
                "structure": {"main_sections": [], "section_hierarchy": {}, "section_titles": {}},
                "themes": {},
                "law_references": {},
                "case_references": {},
                "key_concepts": [],
                "affected_groups": [],
                "legal_exceptions": [],
                "title": "Juridisk vejledning"
            }
    
    def _process_segment(self, segment, context_summary, doc_id, section_id, section_title, options):
        """Processer et segment til chunks med semantisk chunking hvis aktiveret"""
        # 1. Uddrag afsnitsoplysninger
        section_id = section_id or self._extract_section_id(segment)
        if not section_id:
            # Hvis vi ikke kan finde et afsnits-ID, generer et midlertidigt
            section_id = f"unknown_section_{hash(segment[:100])}"
        
        if not section_title:
            section_title = self._extract_section_title(segment, section_id)
        
        # 2. Vælg processeringsmetode baseret på indstillinger
        if hasattr(st.session_state, 'extract_subsections') and st.session_state.extract_subsections:
            # Proces med underafsnit
            return self._process_with_subsections(segment, context_summary, doc_id, section_id, section_title, options)
        elif hasattr(st.session_state, 'semantic_chunking') and st.session_state.semantic_chunking:
            # Brug semantisk chunking
            return self._semantic_chunking(segment, context_summary, doc_id, section_id, section_title, None, options)
        else:
            # Brug standard chunking
            return self._basic_chunking(segment, context_summary, doc_id, section_id, section_title, options)

    def _process_with_subsections(self, segment, context_summary, doc_id, section_id, section_title, options):
        """Processor et segment med opdeling i underafsnit"""
        # Find underafsnit baseret på overskrifter
        subsection_pattern = r'\n([A-Z][a-zæøåA-ZÆØÅ\s]+)(?:\n|$)'
        parts = re.split(subsection_pattern, segment)
        
        chunks = []
        current_subsection = None
        
        # Håndter første del (ofte introduktion før første underafsnit)
        if parts and not re.match(r'^[A-Z][a-zæøåA-ZÆØÅ\s]+$', parts[0].strip()):
            # Dette er introduktionstekst
            intro_text = parts[0].strip()
            if intro_text:
                # Brug semantisk chunking hvis aktiveret
                if hasattr(st.session_state, 'semantic_chunking') and st.session_state.semantic_chunking:
                    intro_chunks = self._semantic_chunking(
                        intro_text, 
                        context_summary, 
                        doc_id, 
                        section_id, 
                        section_title, 
                        "Indhold"  # Standardværdi for introtekst
                    )
                else:
                    intro_chunks = self._create_basic_chunks(
                        intro_text, 
                        context_summary, 
                        doc_id, 
                        section_id, 
                        section_title, 
                        subsection="Indhold"
                    )
                chunks.extend(intro_chunks)
        
        # Gennemløb resten af delene (skiftevis subsection_name og subsection_content)
        for i in range(1, len(parts)):
            if i % 2 == 1:  # Underafsnit-navn
                current_subsection = parts[i].strip()
            else:  # Underafsnit-indhold
                if current_subsection and parts[i].strip():
                    # Process dette underafsnit med semantisk chunking hvis aktiveret
                    if hasattr(st.session_state, 'semantic_chunking') and st.session_state.semantic_chunking:
                        subsection_chunks = self._semantic_chunking(
                            parts[i].strip(), 
                            context_summary, 
                            doc_id, 
                            section_id, 
                            section_title, 
                            current_subsection
                        )
                    else:
                        subsection_chunks = self._create_basic_chunks(
                            parts[i].strip(), 
                            context_summary, 
                            doc_id, 
                            section_id, 
                            section_title, 
                            subsection=current_subsection
                        )
                    chunks.extend(subsection_chunks)
        
        # Hvis vi ikke fandt underafsnit, brug standard chunking
        if not chunks:
            if hasattr(st.session_state, 'semantic_chunking') and st.session_state.semantic_chunking:
                chunks = self._semantic_chunking(segment, context_summary, doc_id, section_id, section_title, None, options)
            else:
                chunks = self._basic_chunking(segment, context_summary, doc_id, section_id, section_title, options)
        
        return chunks

    def _semantic_chunking(self, text, context_summary, doc_id, section_id, section_title, subsection=None, options=None):
        """Semantisk chunking der bevarer juridiske ræsonnementer"""
        chunks = []
        
        # Hvis teksten er meget kort, opret et enkelt chunk
        if len(text.strip()) < getattr(st.session_state, 'min_chunk_size', 250):
            return self._create_single_chunk(text, context_summary, doc_id, section_id, section_title, subsection)
        
        # 1. Uddrag eksempler først hvis aktiveret
        if hasattr(st.session_state, 'extract_examples') and st.session_state.extract_examples:
            example_chunks = self._extract_examples(text, context_summary, doc_id, section_id, section_title, subsection)
            if example_chunks:
                chunks.extend(example_chunks)
                # Fjern eksemplerne fra teksten
                for chunk in example_chunks:
                    text = text.replace(chunk["content"], "")
        
        # 2. Uddrag domsoversigter hvis aktiveret
        if hasattr(st.session_state, 'extract_case_tables') and st.session_state.extract_case_tables and "dom" in text.lower():
            table_chunks = self._extract_case_tables(text, context_summary, doc_id, section_id, section_title, subsection)
            if table_chunks:
                chunks.extend(table_chunks)
                # Fjern tabellerne fra teksten
                for chunk in table_chunks:
                    text = text.replace(chunk["content"], "")
        
        # 3. Identificer semantiske brudpunkter for resten af teksten
        text = text.strip()
        if not text:
            return chunks
        
        # Find semantiske brudpunkter (baseret på juridisk logik)
        segments = self._split_by_semantic_breakpoints(text)
        
        # Behandl hvert semantisk segment
        for segment in segments:
            if segment.strip():
                # Bestemmelse af chunk-type baseret på indhold
                chunk_type = self._determine_chunk_type(segment)
                
                # Opret chunk med den bestemte type
                chunks.append(self._create_chunk(
                    segment.strip(), 
                    context_summary, doc_id, section_id, section_title, subsection,
                    chunk_type=chunk_type, is_example=False
                ))
        
        return chunks
    
    def _determine_chunk_type(self, text):
        """Bestemmer chunk-typen baseret på indhold"""
        text_lower = text.strip().lower()
        
        if re.search(r'^\s*(?:hovedregel|regel)\b', text_lower):
            return "regel"
        elif re.search(r'^\s*bemærk\b', text_lower):
            return "note"
        elif re.search(r'^\s*(?:se også|der henvises til)\b', text_lower):
            return "reference"
        elif re.search(r'^\s*(?:undtagelse|særregel)\b', text_lower):
            return "undtagelse"
        elif re.search(r'^\s*(?:eksempel|for eksempel|til illustration)\b', text_lower):
            return "eksempel"
        elif re.search(r'^\s*(?:definition|defineres som|forstås ved)\b', text_lower):
            return "definition"
        else:
            return "text"
    
    def _split_by_semantic_breakpoints(self, text):
        """Opdeler tekst ved semantiske brudpunkter baseret på juridisk logik"""
        # Hent standard målstørrelse (vil blive justeret per segment baseret på indhold)
        base_target_size = getattr(st.session_state, 'target_chunk_size', 1000)
        
        # Hvis teksten er kortere end målstørrelsen, behold den som ét segment
        if len(text) <= base_target_size:
            return [text]
        
        # Semantiske markører der indikerer nye logiske sektioner
        # Sorteret efter prioritet/styrke af brudpunktet
        primary_markers = [
            r'^\s*Hovedregel\b',
            r'^\s*Regel\b',
            r'^\s*Undtagelse(n|rne)?\b',
            r'^\s*Eksempel\b',
            r'^\s*Definition\b'
        ]
        
        secondary_markers = [
            r'^\s*Se også\b', 
            r'^\s*Der henvises til\b',
            r'^\s*I praksis\b',
            r'^\s*Forudsætninger(ne)?\b',
            r'^\s*Betingelser(ne)?\b'
        ]
        
        tertiary_markers = [
            r'Det (antages|forudsættes|kræves)\b',
            r'Det (bemærkes|fremgår|følger)\b',
            r'Dette gælder (også|dog|ikke)\b',
            r'Følgende (betingelser|krav|forudsætninger)'
        ]
        
        # Identificér potentielle breakpoints
        breakpoints = []
        
        # Håndter først afsnit som klare brudpunkter
        paragraphs = []
        for para in re.split(r'\n\s*\n', text):
            if para.strip():
                paragraphs.append(para)
                
        # Gå igennem paragraffer og identificer brudpunkter
        for i, para in enumerate(paragraphs):
            # Check for primære markører (stærkeste brud)
            for marker in primary_markers:
                if re.search(marker, para, re.MULTILINE):
                    breakpoints.append((i, 10))  # Giv høj vægt (10) til primære markører
                    break
            else:
                # Hvis ingen primære markører blev fundet, check sekundære
                for marker in secondary_markers:
                    if re.search(marker, para, re.MULTILINE):
                        breakpoints.append((i, 5))  # Giv medium vægt (5) til sekundære markører
                        break
                else:
                    # Hvis ingen sekundære markører blev fundet, check tertiære
                    for marker in tertiary_markers:
                        if re.search(marker, para, re.MULTILINE):
                            breakpoints.append((i, 2))  # Giv lav vægt (2) til tertiære markører
                            break
        
        # Hvis vi ikke har nogen brudpunkter, brug standardopdeling (afsnit)
        if not breakpoints:
            # Del på afsnit, men slå sammen hvis de er for små
            current_segment = ""
            segments = []
            for para in paragraphs:
                if len(current_segment) + len(para) > base_target_size * 1.5:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = para
                    else:
                        # Dette afsnit er for stort alene, del det ved sætningsgrænser
                        segments.extend(self._split_by_size(para, target_size=base_target_size))
                else:
                    if current_segment:
                        current_segment += "\n\n" + para
                    else:
                        current_segment = para
            
            if current_segment:
                segments.append(current_segment)
                
            return segments
        
        # Sortér breakpoints efter position
        breakpoints.sort(key=lambda x: x[0])
        
        # Opbyg segmenter
        segments = []
        start_idx = 0
        
        for break_idx, weight in breakpoints:
            if break_idx <= start_idx:
                continue  # Skip hvis vi allerede har brugt dette brudpunkt
                
            # Tilføj paragraffer fra start_idx til break_idx (ekskl.)
            segment_paras = paragraphs[start_idx:break_idx]
            
            # Vurder om det er værd at lave et brud her baseret på størrelse og vægt
            segment_text = "\n\n".join(segment_paras)
            next_para = paragraphs[break_idx] if break_idx < len(paragraphs) else ""
            
            # Hvis segmentet er meget lille, og vægten er lav, overvej at fortsætte
            if len(segment_text) < base_target_size * 0.4 and weight < 5 and len(segment_text + next_para) <= base_target_size:
                continue  # Skip dette brudpunkt og fortsæt til næste
                
            if segment_text:
                segments.append(segment_text)
            
            start_idx = break_idx
        
        # Tilføj resten
        if start_idx < len(paragraphs):
            segments.append("\n\n".join(paragraphs[start_idx:]))
        
        # Tjek størrelsen på segmenterne og juster om nødvendigt
        final_segments = []
        for segment in segments:
            if len(segment) > base_target_size * 1.5:
                # Dette segment er stadig for stort, opdel det yderligere
                subsegments = self._split_by_size(segment, target_size=base_target_size)
                final_segments.extend(subsegments)
            else:
                final_segments.append(segment)
                
        return final_segments
    
    def _split_into_sentences(self, text):
        """Opdeler tekst i sætninger med respekt for juridiske forkortelser"""
        # Definer forkortelser der kan indeholde punktummer, men ikke indikerer sætningsslut
        abbreviations = [
            r'jf\.', r'bl\.a\.', r'f\.eks\.', r'pkt\.', r'nr\.', r'stk\.', 
            r'ca\.', r'evt\.', r'osv\.', r'mv\.', r'inkl\.', r'ekskl\.',
            r'hhv\.', r'vedr\.', r'afd\.', r'div\.', r'pga\.'
        ]
        
        # Erstat forkortelser midlertidigt for at undgå forkert opdeling
        for abbr in abbreviations:
            text = re.sub(abbr, abbr.replace('.', '<DOT>'), text)
        
        # Del ved sætningsgrænser
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÆØÅ])', text)
        
        # Gendan forkortelser
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        return sentences
    
    def _split_by_size(self, text, target_size=None):
        """Opdeler tekst i chunks af målstørrelse med respekt for sætningsgrænser"""
        if target_size is None:
            target_size = getattr(st.session_state, 'target_chunk_size', 1000)
            
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Hvis sentence alene er større end målstørrelsen*1.5, del det yderligere
            if len(sentence) > target_size * 1.5:
                # Del ved kommaer eller andre naturlige pauser
                clause_splits = re.split(r'(?<=[,:])\s+', sentence)
                
                for clause in clause_splits:
                    if len(current_chunk + clause + " ") <= target_size:
                        current_chunk += clause + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = clause + " "
            # Ellers forsøg at holde sætninger sammen
            elif len(current_chunk + sentence + " ") <= target_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        # Tilføj sidste chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _extract_content_by_pattern(self, text, patterns, context_summary, doc_id, section_id, 
                                   section_title, subsection, chunk_type, is_example=False):
        """Generisk metode til at udtrække indhold baseret på mønstre"""
        chunks = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                content = match.group(1).strip()
                if content:
                    # Udled metadata for indholdet
                    metadata = self._extract_metadata_for_content(content, chunk_type)
                    
                    # Opret chunk for dette indhold
                    chunk = self._create_chunk(
                        content, 
                        context_summary, 
                        doc_id, 
                        section_id, 
                        section_title, 
                        subsection,
                        chunk_type=chunk_type, 
                        is_example=is_example,
                        **metadata
                    )
                    
                    chunks.append(chunk)
                    
        return chunks

    def _extract_examples(self, text, context_summary, doc_id, section_id, section_title, subsection):
        """Udtrækker eksempler fra teksten og tilføjer kontekst til hovedregel"""
        # Definér mønstre for eksempler
        example_patterns = [
            r'(Eksempel\s+\d+\s*[:\.][^E]*?)(?=Eksempel\s+\d+|$)',  # Standard nummererede eksempler
            r'(Eksempel\s*[:\.][^E]*?)(?=Eksempel|$)',              # Generelle eksempler uden nummer
            r'(Følgende\s+eksempel[^\.]*?illustrerer.*?(?:\n\n|$))',  # Følgende eksempel illustrerer...
            r'((?:Til\s+illustration|Som\s+eksempel)[^\.]*?(?:kan\s+nævnes|vises).*?(?:\n\n|$))',  # Til illustration/Som eksempel
            r'(For\s+eksempel\s+(?:kan|vil|har).*?(?:\n\n|$))'        # For eksempel...
        ]
        
        chunks = []
        example_count = 0
        
        for pattern in example_patterns:
            examples = re.finditer(pattern, text, re.DOTALL)
            
            for match in examples:
                example_text = match.group(1).strip()
                if example_text:
                    example_count += 1
                    
                    # Udled eksempel-nummer hvis det findes
                    example_num_match = re.search(r'Eksempel\s+(\d+)', example_text)
                    example_num = example_num_match.group(1) if example_num_match else str(example_count)
                    
                    # Find domsreferencer i eksemplet
                    case_refs = self._extract_case_refs_from_text(example_text)
                    
                    # Find lovhenvisninger i eksemplet
                    structured_refs, primary_law_ref = self._extract_law_refs_from_text(example_text)
                    
                    # Udled persongrupper
                    affected_groups = self._extract_affected_groups(example_text)
                    
                    # Udled juridiske undtagelser
                    legal_exceptions = self._extract_legal_exceptions(example_text)
                    
                    # Find kontekst for eksemplet (relaterede regler)
                    related_rule = self._find_related_rule(text, example_text)
                    
                    # Opret chunk for dette eksempel
                    example_chunk = self._create_chunk(
                        example_text, 
                        context_summary, 
                        doc_id, 
                        section_id, 
                        section_title, 
                        subsection,
                        chunk_type="eksempel", 
                        is_example=True,  # Altid sæt is_example til True for eksempler
                        example_num=example_num,
                        case_references=case_refs,
                        law_references=structured_refs,
                        affected_groups=affected_groups,
                        legal_exceptions=legal_exceptions
                    )
                    
                    # Tilføj relateret regel til metadata
                    if related_rule:
                        example_chunk["metadata"]["related_rule"] = related_rule
                    
                    # Tilføj primær lovreference
                    if primary_law_ref:
                        example_chunk["metadata"]["primary_law_ref"] = primary_law_ref
                    
                    chunks.append(example_chunk)
        
        # Identificer også implicitte eksempler (casebaserede beskrivelser uden "eksempel"-markør)
        implicit_example_pattern = r'(?<!\w)((?:Hvis|Lad os antag|Tænk på)[^\.]*?(?:person|firma|virksomhed|selskab)[^\.]*?(?:der|som|hvilket)[^\.]*?(?:\n\n|$))'
        implicit_examples = re.finditer(implicit_example_pattern, text, re.DOTALL)
        
        implicit_count = 0
        for match in implicit_examples:
            implicit_example_text = match.group(1).strip()
            if implicit_example_text and len(implicit_example_text) > 100:  # Undgå korte matches
                implicit_count += 1
                
                # Håndter som et eksempel
                case_refs = self._extract_case_refs_from_text(implicit_example_text)
                structured_refs, primary_law_ref = self._extract_law_refs_from_text(implicit_example_text)
                affected_groups = self._extract_affected_groups(implicit_example_text)
                legal_exceptions = self._extract_legal_exceptions(implicit_example_text)
                
                # Find kontekst for eksemplet
                related_rule = self._find_related_rule(text, implicit_example_text)
                
                # Opret chunk for dette implicitte eksempel
                implicit_chunk = self._create_chunk(
                    implicit_example_text, 
                    context_summary, 
                    doc_id, 
                    section_id, 
                    section_title, 
                    subsection,
                    chunk_type="eksempel", 
                    is_example=True,  # Marker som eksempel selvom det er implicit
                    example_num=f"i{implicit_count}",  # Præfiks 'i' for implicit
                    case_references=case_refs,
                    law_references=structured_refs,
                    affected_groups=affected_groups,
                    legal_exceptions=legal_exceptions
                )
                
                if related_rule:
                    implicit_chunk["metadata"]["related_rule"] = related_rule
                
                if primary_law_ref:
                    implicit_chunk["metadata"]["primary_law_ref"] = primary_law_ref
                    
                chunks.append(implicit_chunk)
        
        return chunks
    
    def _find_related_rule(self, full_text, example_text):
        """Finder den relaterede regel til et eksempel"""
        # Forsøg at finde tekst før eksemplet der indeholder en regel
        example_start = full_text.find(example_text)
        if example_start <= 0:
            return None
            
        # Find teksten før eksemplet
        text_before = full_text[:example_start].strip()
        
        # Check for specifikke rege-lignende afsnit
        rule_patterns = [
            r'(?:Hovedregel|Regel).*?(?=\n\n)',
            r'(?:Reglerne|Reglen).*?(?=\n\n)',
            r'(?:Efter|Ifølge).*?(?:gælder|er).*?(?=\n\n)'
        ]
        
        for pattern in rule_patterns:
            rule_matches = list(re.finditer(pattern, text_before, re.DOTALL))
            if rule_matches:
                # Tag den sidste match (den nærmeste regel før eksemplet)
                rule_text = rule_matches[-1].group(0).strip()
                # Begræns længden for at holde den kompakt
                max_rule_length = 300
                if len(rule_text) > max_rule_length:
                    rule_text = rule_text[:max_rule_length] + "..."
                return rule_text
        
        # Hvis vi ikke finder en regel, prøv at tage det sidste afsnit før eksemplet
        paragraphs = text_before.split("\n\n")
        if paragraphs:
            last_paragraph = paragraphs[-1].strip()
            # Kun returner hvis det ser ud til at være relevant
            if len(last_paragraph) > 30 and len(last_paragraph) < 300:
                return last_paragraph
                
        return None

    def _extract_law_refs_from_text(self, text):
        """Udtrækker strukturerede lovhenvisninger fra tekst med dynamiske forkortelser"""
        law_refs = []
    
        # Find lovhenvisninger med lov + § + paragraf
        law_pattern = r'([a-zæøåA-ZÆØÅ]+(?:lovens?|loven))\s+§[§]?\s*(\d+\s*[A-Za-z]?(?:\s*[-–]\s*\d+\s*[A-Za-z]?)?)'
        for match in re.finditer(law_pattern, text, re.IGNORECASE):
            lov_text = match.group(1).lower()
            paragraf_range = match.group(2).strip()
            
            # Bestem forkortelse baseret på lovnavn ved hjælp af domænekonfigurationen
            prefix = None
            for lovnavn, forkortelse in self.law_abbreviations.items():
                if lovnavn.lower() in lov_text:
                    prefix = forkortelse
                    break
                    
            # Hvis vi ikke finder en konfigureret forkortelse, brug første bogstav af hvert ord
            if not prefix:
                words = lov_text.replace("lovens", "").replace("loven", "").strip().split()
                prefix = "".join(word[0].upper() for word in words if word)
                if not prefix:
                    prefix = lov_text  # Sidste udvej: brug hele lovnavnet
                
            # Håndter paragraf-ranges (f.eks. §§ 4-6)
            if '-' in paragraf_range or '–' in paragraf_range:
                range_match = re.search(r'(\d+)\s*[-–]\s*(\d+)', paragraf_range)
                if range_match:
                    start_num = int(range_match.group(1))
                    end_num = int(range_match.group(2))
                    for num in range(start_num, end_num + 1):
                        ref = f"{prefix} § {num}"
                        law_refs.append(ref)
            else:
                # Udled stykke hvis det findes
                stykke_match = re.search(r'(?:,?\s*(?:stk\.|stykke)\s*(\d+))', text)
                stykke = stykke_match.group(1) if stykke_match else ""
                
                ref = f"{prefix} § {paragraf_range}"
                if stykke:
                    ref += f", stk. {stykke}"
                    
                law_refs.append(ref)
        
        # Find direkte paragrafhenvisninger (§ 33 A eller §§ 4-6)
        direct_pattern = r'(§[§]?\s*\d+\s*[A-Za-z]?(?:\s*[-–]\s*\d+\s*[A-Za-z]?)?)'
        for match in re.finditer(direct_pattern, text):
            # Håndter paragraf-ranges for direkte referencer
            direct_ref = match.group(1)
            
            if '§§' in direct_ref and ('-' in direct_ref or '–' in direct_ref):
                range_match = re.search(r'§§\s*(\d+)\s*[-–]\s*(\d+)', direct_ref)
                if range_match:
                    start_num = int(range_match.group(1))
                    end_num = int(range_match.group(2))
                    
                    # Find mulig lovkontekst i nærheden af referencen
                    context_before = text[max(0, text.find(direct_ref)-50):text.find(direct_ref)]
                    context_after = text[text.find(direct_ref)+len(direct_ref):min(len(text), text.find(direct_ref)+len(direct_ref)+50)]
                    
                    # Find hvilken lov der refereres til baseret på kontekst
                    prefix = self._determine_law_from_context(context_before + context_after)
                    
                    for num in range(start_num, end_num + 1):
                        ref = f"{prefix} § {num}"
                        if ref not in law_refs:  # Undgå dubletter
                            law_refs.append(ref)
            else:
                # Udled paragraf og stykke
                para_match = re.search(r'§\s*(\d+\s*[A-Za-z]?)', direct_ref)
                stykke_match = re.search(r'(?:,?\s*(?:stk\.|stykke)\s*(\d+))', text)
                
                if para_match:
                    paragraf = para_match.group(1).strip()
                    stykke = stykke_match.group(1) if stykke_match else ""
                    
                    # Find mulig lovkontekst i nærheden af referencen
                    context_before = text[max(0, text.find(direct_ref)-50):text.find(direct_ref)]
                    context_after = text[text.find(direct_ref)+len(direct_ref):min(len(text), text.find(direct_ref)+len(direct_ref)+50)]
                    
                    # Find hvilken lov der refereres til baseret på kontekst
                    prefix = self._determine_law_from_context(context_before + context_after)
                    
                    ref = f"{prefix} § {paragraf}"
                    if stykke:
                        ref += f", stk. {stykke}"
                    
                    if ref not in law_refs:  # Undgå dubletter
                        law_refs.append(ref)
        
        # Tæl forekomster af hver reference for at finde primær reference
        ref_counts = {}
        for ref in law_refs:
            ref_counts[ref] = text.lower().count(ref.lower())
        
        # Find den mest omtalte reference som primær
        primary_ref = None
        if ref_counts:
            primary_ref = max(ref_counts.items(), key=lambda x: x[1])[0]
        
        # Konvertér til struktureret format
        structured_refs = []
        for ref in law_refs:
            structured_refs.append({
                "ref": ref,
                "is_primary": ref == primary_ref
            })
        
        return structured_refs, primary_ref
    
    def _determine_law_from_context(self, context):
        """Bestemmer hvilken lov der refereres til baseret på kontekst"""
        context_lower = context.lower()
        
        # Tjek først for lovnavne i konteksten
        for lovnavn, forkortelse in self.law_abbreviations.items():
            if lovnavn.lower() in context_lower:
                return forkortelse
                
        # Tjek for forkortelser i konteksten
        for forkortelse in self.law_abbreviations.values():
            if forkortelse.lower() in context_lower:
                return forkortelse
                
        # Returner standardværdi hvis ingen lov kunne bestemmes
        # Dette kunne være den mest almindelige lov i dokumentet, bestemt fra domain_config
        if hasattr(self, 'domain_config') and self.domain_config.get('primary_law_prefix'):
            return self.domain_config['primary_law_prefix']
            
        # Sidste udvej: returner "LOV" som generisk forkortelse
        return "LOV"

    def _extract_case_refs_from_text(self, text):
        """Udtrækker domsreferencer fra tekst"""
        case_refs = []
        
        # Find dynamiske mønstre baseret på retsområde og danske domstole
        case_patterns = [
            # Domme fra Højesteret, Landsretten, Sø- og Handelsretten, etc.
            r'((?:UfR|U|TfS|FM|MAD)\s*\d{4}[.,]\s*\d+(?:\s*[A-ZØ]+)?)',
            r'((?:Højesterets|Landsrettens|Sø-\s*og\s*Handelsrettens)\s*dom\s*af\s*\d{1,2}\.\s*\w+\s*\d{4})',
            # Administrative afgørelser 
            r'((?:SKM|LSR|TfS|TSS)[-\s]*\d{4}[.,]\s*\d+(?:\s*[A-ZØ]+)?)',
            # Andre formater
            r'([A-ZÆØÅ]{2,5}\s*\d{4}[-.,/]\s*\d+(?:\s*[A-ZØ]+)?)'
        ]
        
        for pattern in case_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                case_ref = match.group(1).strip()
                if case_ref and case_ref not in case_refs:
                    case_refs.append(case_ref)
        
        return case_refs

    def _extract_affected_groups(self, text):
        """Udtrækker berørte persongrupper fra teksten dynamisk fra domæne-konfigurationen"""
        affected_groups = []
        
        # Brug domænekonfigurationen til at finde persongrupper
        if hasattr(self, 'person_groups') and self.person_groups:
            for group, keywords in self.person_groups.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                        affected_groups.append(group)
                        break  # Kun tilføj gruppen én gang
        
        # Særlige mønstre for at finde persongrupper
        patterns = [
            r'(?:for|gælder for|omfatter)\s+([^\.;,]+?)\s+(?:der|som|når)',
            r'(?:personer|ydere|pligtige|borgere|virksomheder)\s+(?:der|som)\s+([^\.;,]+)',
            r'([^\.;,]+)\s+(?:er|kan være|anses for)\s+(?:pligtig|omfattet|forpligtet|berettiget)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                group = match.group(1).strip()
                # Rens gruppen for støjord
                group = re.sub(r'\b(personer|ydere|pligtige|alle|disse|de|bestemte)\b', '', group).strip()
                # Undgå for korte eller lange udtryk
                if len(group) > 5 and len(group) < 50:
                    affected_groups.append(group)
        
        # Fjern duplikater men behold rækkefølgen
        unique_groups = []
        for group in affected_groups:
            if group not in unique_groups:
                unique_groups.append(group)
        
        return unique_groups

    def _extract_legal_exceptions(self, text):
        """Udtrækker juridiske undtagelser og specialregler"""
        exceptions = []
        
        # Mønstre der kan indikere undtagelser
        exception_patterns = [
            r'(?:undtagelse|særregel|specialregel)[^\.;,]*?(?=\.|;|$)',
            r'(?:gælder ikke|finder ikke anvendelse)[^\.;,]*?(?=\.|;|$)',
            r'(?:medmindre|dog ikke|undtaget herfra er)[^\.;,]*?(?=\.|;|$)',
            r'(?:uanset|til trods for)[^\.;,]*?(?=\.|;|$)',
            r'(?:Hovedreglen|Udgangspunktet).*?(?:men|dog)[^\.;,]*?(?=\.|;|$)'
        ]
        
        for pattern in exception_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                exception = match.group(0).strip()
                if exception and len(exception) > 10:  # Undgå for korte udtryk
                    exceptions.append(exception)
        
        # Tilføj domænespecifikke undtagelser fra domænekonfigurationen
        if hasattr(self, 'domain_config') and 'legal_exceptions' in self.domain_config:
            for exception in self.domain_config['legal_exceptions']:
                if exception.lower() in text.lower() and exception not in exceptions:
                    exceptions.append(exception)
        
        # Fjern duplikater
        unique_exceptions = []
        for exc in exceptions:
            # Normalisér til lowercase for sammenligning
            norm_exc = exc.lower()
            if not any(norm_exc == e.lower() for e in unique_exceptions):
                unique_exceptions.append(exc)
        
        return unique_exceptions

    def _extract_concepts(self, text, themes=None):
        """Udtrækker dynamiske nøglekoncepter fra teksten baseret på dokumentets indhold"""
        # Start med eventuelle kendte temaer
        if themes and isinstance(themes, list):
            concepts = themes[:3]
        else:
            concepts = []
    
        # Hvis vi har context_summary med domæne-specifik viden
        if hasattr(self, 'domain_config') and 'key_concepts' in self.domain_config:
            # Hent nøglekoncepter identificeret under dokumentanalysen
            domain_concepts = self.domain_config.get('key_concepts', [])
        
            # Tjek for hvert domæne-koncept om det findes i teksten
            for concept in domain_concepts:
                if concept.lower() in text.lower() and concept not in concepts:
                    concepts.append(concept)
    
        # Find definitioner direkte i teksten (dynamisk)
        definition_patterns = [
            r'Ved\s+([^\.;,]+)\s+forstås',
            r'([^\.;,]+)\s+defineres\s+som',
            r'([^\.;,]+)\s+betyder\s+i\s+denne\s+sammenhæng'
        ]
    
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip().lower()
                term = re.sub(r'\b(herved|således|dermed|hermed|at)\b', '', term).strip()
                if len(term) > 3 and len(term) < 50 and term not in concepts:
                    concepts.append(term)
    
        # Hvis vi stadig har for få koncepter, brug GPT-4o til at identificere flere
        if len(concepts) < 3 and len(text) > 100:
            # Bruger cached_call_gpt4o for at undgå at kalde API'et for mange gange
            prompt = f"""
            Identificer op til 5 juridiske nøglebegreber i denne tekst.
            Giv kun begreberne, ét pr. linje.
        
            Tekst: {text[:1000]}
            """
            try:
                result = cached_call_gpt4o(prompt, model="gpt-4o", json_mode=False)
                if result:
                    # Analyser resultatet (antager at hvert begreb er på sin egen linje)
                    for line in result.strip().split('\n'):
                        concept = line.strip()
                        if concept and concept not in concepts:
                            concepts.append(concept)
            except Exception as e:
                self.logger.warning(f"Fejl ved kald til GPT-4o for konceptudtrækning: {e}")
    
        # Begræns til maks 7 koncepter
        return concepts[:7]
    
    def _determine_complexity(self, text, law_refs, case_refs):
        """Bestemmer kompleksiteten af et chunk"""
        # Simpel scoring-mekanisme
        complexity_score = 0
        # Længde-baseret kompleksitet
        text_length = len(text)
        if text_length > 1000:
            complexity_score += 2
        elif text_length > 500:
            complexity_score += 1
        
        # Antal lovhenvisninger
        if isinstance(law_refs, list):
            if all(isinstance(item, dict) for item in law_refs):
                # Strukturerede referencer
                law_count = len(law_refs)
            else:
                # Ustrukturerede referencer
                law_count = len(law_refs)
                
            if law_count > 3:
                complexity_score += 2
            elif law_count > 1:
                complexity_score += 1
        
        # Antal domsreferencer
        if len(case_refs) > 2:
            complexity_score += 2
        elif len(case_refs) > 0:
            complexity_score += 1
        
        # Lingvistiske kompleksitetsmarkører
        complex_terms = ["dog", "medmindre", "såfremt", "forudsat", "betinget af", "undtagelsesvis"]
        for term in complex_terms:
            if term in text.lower():
                complexity_score += 1
                
        # Konvertér score til kategorier
        if complexity_score > 4:
            return "kompleks"
        elif complexity_score > 2:
            return "moderat"
        else:
            return "simpel"

    def _extract_question_types(self, text):
        """Identificerer relevante spørgsmålstyper baseret på tekstindholdet"""
        found_types = []
        
        # Tjek hvert spørgsmålsmønster fra domænekonfigurationen
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    found_types.append(q_type)
                    break  # Gå videre til næste spørgsmålstype når vi har et match
        
        # Hvis vi ikke fandt nogen spørgsmålstyper ved mønstre, anvend AI til at finde relevante typer
        if not found_types and len(text) > 200:
            try:
                # Brug AI til at identificere relevante spørgsmålstyper
                prompt = f"""
                Denne tekst beskriver juridiske regler. Hvilke typer af spørgsmål ville være relevante at stille til denne tekst?
                Vælg 1-3 kategorier fra følgende liste:
                {", ".join(self.question_patterns.keys())}
                
                Svar kun med kategorierne adskilt af komma.
                
                Tekst: {text[:1000]}
                """
                
                result = cached_call_gpt4o(prompt, model="gpt-4o", json_mode=False)
                if result:
                    ai_types = [t.strip() for t in result.split(',')]
                    # Filtrer for at sikre vi kun inkluderer gyldige typer
                    found_types = [t for t in ai_types if t in self.question_patterns]
            except Exception as e:
                self.logger.warning(f"Fejl ved bestemmelse af spørgsmålstyper: {e}")
                # Fortsæt uden AI-genererede typer
        
        return found_types

    def _metadata_extractor(self, text, context_summary):
        """Centraliseret metadata-udtrækning fra tekst"""
        # Udled lovhenvisninger
        structured_refs, primary_law_ref = self._extract_law_refs_from_text(text)
        
        # Udled domsreferencer
        case_references = self._extract_case_refs_from_text(text)
        
        # Udled persongrupper
        affected_groups = self._extract_affected_groups(text)
        
        # Udled juridiske undtagelser
        legal_exceptions = self._extract_legal_exceptions(text)
        
        # Udled temaer fra context_summary
        if context_summary and "key_concepts" in context_summary:
            themes = context_summary["key_concepts"]
        else:
            themes = self.default_themes
        
        # Udled koncepter baseret på indhold og kontekst
        concepts = self._extract_concepts(text, themes)
        
        # Bestem kompleksitet
        complexity = self._determine_complexity(text, structured_refs, case_references)
        
        # Identificer spørgsmålstyper baseret på indhold
        question_types = self._extract_question_types(text)
        
        return {
            "law_references": structured_refs,
            "primary_law_ref": primary_law_ref,
            "case_references": case_references,
            "affected_groups": affected_groups,
            "legal_exceptions": legal_exceptions,
            "concepts": concepts,
            "complexity": complexity,
            "question_types": question_types
        }

    def _create_chunk(self, text, context_summary, doc_id, section_id, section_title, subsection, 
                     chunk_type="text", is_example=False, example_num=None, **kwargs):
        """Opret et chunk med metadata"""
        if not text.strip():
            return None
        
        # Udled sti i hierarkiet
        hierarchy_path = self._get_hierarchy_path(section_id, context_summary)
        
        # Udled temaer
        themes = self._get_themes_for_section(section_id, context_summary)
        
        # Udled metadata hvis ikke allerede givet
        if not kwargs:
            metadata_dict = self._metadata_extractor(text, context_summary)
        else:
            metadata_dict = kwargs
        
        # Generer chunk ID
        chunk_id = f"{section_id}_{subsection}_{example_num or chunk_type}_{hash(text[:50])}"
        
        # Opret chunk
        chunk = {
            "content": text,
            "metadata": {
                "doc_id": doc_id,
                "doc_type": "juridisk_vejledning",
                "version_date": context_summary.get("version_date", ""),
                
                "section": section_id,
                "section_title": section_title if section_title else f"Afsnit {section_id}",
                "subsection": subsection,
                
                "hierarchy_path": hierarchy_path,
                "hierarchy_level": len(hierarchy_path) if hierarchy_path else 1,
                
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "is_example": is_example,
                
                "concepts": metadata_dict.get("concepts", []),
                "theme": themes[0] if themes else "",
                "subtheme": themes[1] if len(themes) > 1 else "",
                
                "law_references": metadata_dict.get("law_references", []),
                "case_references": metadata_dict.get("case_references", []),
                "affected_groups": metadata_dict.get("affected_groups", []),
                "legal_exceptions": metadata_dict.get("legal_exceptions", []),
                
                "complexity": metadata_dict.get("complexity", "simpel"),
                "authority": 1,  # Højeste autoritetsgrad (Juridisk Vejledning)
                "question_types": metadata_dict.get("question_types", [])
            }
        }
        
        # Tilføj primær lovreference til metadata hvis den findes
        if "primary_law_ref" in metadata_dict and metadata_dict["primary_law_ref"]:
            chunk["metadata"]["primary_law_ref"] = metadata_dict["primary_law_ref"]
        
        # Tilføj eksempel-specifik metadata
        if is_example and example_num:
            chunk["metadata"]["example_num"] = example_num
        
        # Tilføj relateret regel til eksempler hvis relevant
        if is_example and "related_rule" in metadata_dict:
            chunk["metadata"]["related_rule"] = metadata_dict["related_rule"]
        
        return chunk

    def _get_target_size_for_chunk_type(self, chunk_type):
        """Bestemmer målstørrelsen for en chunk baseret på indholdstypen"""
        # Sæt standardstørrelse fra session state
        base_size = getattr(st.session_state, 'target_chunk_size', 1000)
        
        # Juster baseret på chunk-type
        if chunk_type == "eksempel":
            # Eksempler behøver ofte ikke at være så store
            return int(base_size * 0.7)  # 70% af standard
        elif chunk_type == "oversigt":
            # Oversigter (som domsoversigter) må gerne være større
            return int(base_size * 1.5)  # 150% af standard
        elif chunk_type == "reference":
            # Referencer (som "Se også"-afsnit) kan være mindre
            return int(base_size * 0.5)  # 50% af standard
        elif chunk_type == "note":
            # Noter (som "Bemærk"-afsnit) kan være mindre
            return int(base_size * 0.6)  # 60% af standard
        elif chunk_type == "regel":
            # Regler må gerne være større for at bevare kontekst
            return int(base_size * 1.3)  # 130% af standard
        elif chunk_type == "definition":
            # Definitioner må gerne være præcise
            return int(base_size * 0.8)  # 80% af standard
        elif chunk_type == "undtagelse":
            # Undtagelser må gerne være større for at forstå konteksten
            return int(base_size * 1.2)  # 120% af standard
        else:
            # Standardstørrelse for andre typer
            return base_size

    def _calculate_retrievability_enhanced(self, text, chunk_type, law_references, case_references, concepts):
        """Beregner en forbedret retrievability score mellem 0-1 for en chunk"""
        score = 0.5  # Standardscore
    
        # Faktor baseret på længde
        length = len(text)
        min_chunk_size = getattr(st.session_state, 'min_chunk_size', 250)
    
        # Brug dynamisk målstørrelse baseret på chunk-type
        target_size = self._get_target_size_for_chunk_type(chunk_type)
    
        if min_chunk_size <= length <= target_size * 1.2:
            score += 0.2  # Optimal længde
        elif length > target_size * 1.5:
            score -= 0.1  # For lang, men mindre straf end tidligere
        elif length < min_chunk_size * 0.8:
            score -= 0.1  # For kort
    
        # Faktor baseret på juridisk indhold
        if law_references:
            if isinstance(law_references, list):
                if all(isinstance(item, dict) for item in law_references):
                    # Strukturerede referencer
                    law_count = len(law_references)
                else:
                    # Ustrukturerede referencer
                    law_count = len(law_references)
                
                score += min(0.2, 0.05 * law_count)  # Op til 0.2 for mange lovhenvisninger
    
        if case_references:
            case_count = len(case_references)
            score += min(0.15, 0.05 * case_count)  # Op til 0.15 for mange domshenvisninger
    
        # Faktor baseret på koncepter
        if concepts:
            score += min(0.15, 0.03 * len(concepts))  # Op til 0.15 for mange koncepter
    
        # Faktor baseret på tekst-kvalitet
        if re.search(r'\bbestår af\b|\bdefineres som\b|\bforståes ved\b|\bfølger af\b', text, re.IGNORECASE):
            score += 0.1  # Definitioner og forklaringer er vigtige
    
        # Faktor baseret på indholdstype
        if chunk_type in ["regel", "definition"]:
            score += 0.15  # Regler og definitioner er meget relevante
        elif chunk_type in ["eksempel", "undtagelse"]:
            score += 0.1  # Eksempler og undtagelser er også relevante
    
        # Normaliser score til 0-1 interval
        return max(0, min(1, score))

    def _basic_chunking(self, segment, context_summary, doc_id, section_id, section_title, options):
        """Brug grundlæggende chunking til at opdele et segment"""
        return self._create_basic_chunks(segment, context_summary, doc_id, section_id, section_title, None)

    def _create_basic_chunks(self, text, context_summary, doc_id, section_id, section_title, subsection=None):
        """Opret grundlæggende chunks fra tekst"""
        chunks = []
    
        # 0. Håndter meget korte tekster
        if len(text.strip()) < getattr(st.session_state, 'min_chunk_size', 250):
            return self._create_single_chunk(text, context_summary, doc_id, section_id, section_title, subsection)
        
        # Tjek først om hele teksten er et eksempel
        example_identifiers = ["eksempel", "til illustration", "som et eksempel", "for eksempel"]
        if any(identifier in text.lower()[:100] for identifier in example_identifiers):
            # Check om det er et eksempel ved at se på de første 100 tegn
            is_example = True
        else:
            is_example = False
        
        # 1. Uddrag eksempler først hvis aktiveret
        if hasattr(st.session_state, 'extract_examples') and st.session_state.extract_examples:
            example_chunks = self._extract_examples(text, context_summary, doc_id, section_id, section_title, subsection)
            if example_chunks:
                chunks.extend(example_chunks)
                # Fjern eksemplerne fra teksten
                for chunk in example_chunks:
                    text = text.replace(chunk["content"], "")
        
            # 2. Uddrag domsoversigter hvis aktiveret
            if hasattr(st.session_state, 'extract_case_tables') and st.session_state.extract_case_tables and "dom" in text.lower():
                table_chunks = self._extract_case_tables(text, context_summary, doc_id, section_id, section_title, subsection)
                if table_chunks:
                    chunks.extend(table_chunks)
                    # Fjern tabellerne fra teksten
                    for chunk in table_chunks:
                        text = text.replace(chunk["content"], "")
        
            # 3. Del resten af teksten i semantiske chunks
            text = text.strip()
            if text:
                if "Se også" in text and len(text) < 500:
                    # Håndter korte "Se også" afsnit som ét chunk
                    chunks.append(self._create_chunk(
                        text, context_summary, doc_id, section_id, section_title, subsection, 
                        chunk_type="reference", is_example=False
                    ))
                elif "Bemærk" in text and len(text) < 500:
                    # Håndter korte "Bemærk" afsnit som ét chunk
                    chunks.append(self._create_chunk(
                        text, context_summary, doc_id, section_id, section_title, subsection, 
                        chunk_type="note", is_example=False
                    ))
                else:
                    # Del i semantiske chunks ved afsnit
                    paragraphs = re.split(r'\n\s*\n', text)
                    
                    for para in paragraphs:
                        if para.strip():
                            # Tjek om dette afsnit er for langt og skal deles yderligere
                            target_size = getattr(st.session_state, 'target_chunk_size', 1000)
                            if len(para) > target_size:
                                # Del i mindre afsnit ved sætningsgrænser
                                sentences = self._split_into_sentences(para)
                                current_chunk = ""
                                
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) < target_size:
                                        current_chunk += sentence + " "
                                    else:
                                        if current_chunk.strip():
                                            chunks.append(self._create_chunk(
                                                current_chunk.strip(), 
                                                context_summary, doc_id, section_id, section_title, subsection,
                                                chunk_type="text", is_example=False
                                            ))
                                        current_chunk = sentence + " "
                                
                                # Tilføj sidste chunk
                                if current_chunk.strip():
                                    chunks.append(self._create_chunk(
                                        current_chunk.strip(), 
                                        context_summary, doc_id, section_id, section_title, subsection,
                                        chunk_type="text", is_example=False
                                    ))
                            else:
                                # Tilføj dette afsnit som ét chunk
                                chunks.append(self._create_chunk(
                                    para, context_summary, doc_id, section_id, section_title, subsection,
                                    chunk_type="text", is_example=False
                                ))
        
        # Tilføj en ekstra check for is_example baseret på indhold
        for chunk in chunks:
            if not chunk["metadata"]["is_example"]:  # Hvis ikke allerede markeret som eksempel
                content = chunk["content"].lower()
                if (content.startswith("eksempel") or 
                    "for eksempel" in content[:100] or 
                    "som eksempel" in content[:100] or 
                    "til illustration" in content[:100]):
                    chunk["metadata"]["is_example"] = True
                    chunk["metadata"]["chunk_type"] = "eksempel"         
        
        return chunks

    def _create_single_chunk(self, text, context_summary, doc_id, section_id, section_title, subsection):
        """Opret et enkeltstående chunk for korte tekster"""
        if not text.strip():
            return []
            
        return [self._create_chunk(
            text, context_summary, doc_id, section_id, section_title, subsection,
            chunk_type="text", is_example=False
        )]

    def _extract_case_tables(self, text, context_summary, doc_id, section_id, section_title, subsection):
        """Udtrækker tabeller med domme og afgørelser"""
        chunks = []
        
        # Find tabeller med domme
        table_patterns = [
            r'((?:Skemaet|Oversigten)\s+viser[\s\S]*?(?=\n\n|$))',
            r'((?:Følgende|Nedenstående)\s+(?:afgørelser|domme|kendelser)[\s\S]*?(?=\n\n|$))',
            r'((?:Domsoversigt|Afgørelsesoversigt)[\s\S]*?(?=\n\n|$))'
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                table_text = match.group(1).strip()
                if table_text:
                    # Find alle domsreferencer i tabellen
                    case_refs = self._extract_case_refs_from_text(table_text)
                    
                    # Hvis vi faktisk fandt domsreferencer, opret chunk
                    if case_refs:
                        # Opret chunk for denne tabel
                        chunks.append(self._create_chunk(
                            table_text, 
                            context_summary, 
                            doc_id, 
                            section_id, 
                            section_title, 
                            subsection or "Oversigt over domme, kendelser, afgørelser mv.",
                            chunk_type="oversigt", 
                            is_example=False,
                            case_references=case_refs
                        ))
        
        return chunks

    def _get_hierarchy_path(self, section_id, context_summary):
        """Udleder hierarkisk sti baseret på afsnits-ID"""
        if not section_id:
            return []
            
        # Del afsnits-ID i komponenter (A.B.1.2.3 -> ["A.B.1", "A.B.1.2", "A.B.1.2.3"])
        parts = section_id.split('.')
        path = []
        
        if len(parts) >= 3:  # A.B.1.2.3
            path.append(f"{parts[0]}.{parts[1]}.{parts[2]}")  # A.B.1
            
        if len(parts) >= 4:  # A.B.1.2.3
            path.append(f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}")  # A.B.1.2
            
        path.append(section_id)  # A.B.1.2.3
        
        return path

    def _get_themes_for_section(self, section_id, context_summary):
        """Udleder temaer for et afsnit fra kontekst"""
        if not section_id:
            return self.default_themes
            
        # Søg efter temaer for dette afsnit
        if context_summary and "themes" in context_summary and section_id in context_summary["themes"]:
            return context_summary["themes"][section_id]
            
        # Søg efter temaer for forældreafsnit
        if context_summary and "themes" in context_summary:
            parts = section_id.split('.')
            if len(parts) >= 4:  # A.B.1.2.3
                parent_id = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}"  # A.B.1.2
                if parent_id in context_summary["themes"]:
                    return context_summary["themes"][parent_id]
                
                # Prøv endnu et niveau op
                if len(parts) >= 3:
                    parent_id = f"{parts[0]}.{parts[1]}.{parts[2]}"  # A.B.1
                    if parent_id in context_summary["themes"]:
                        return context_summary["themes"][parent_id]
        
        # Brug generelle temaer fra dokumentet
        if context_summary and "key_concepts" in context_summary and context_summary["key_concepts"]:
            # Konverter til liste hvis det er en streng
            if isinstance(context_summary["key_concepts"], str):
                return [context_summary["key_concepts"]]
            return context_summary["key_concepts"][:2]  # Brug de to første generelle temaer
            
        # Default temaer
        return self.default_themes

    def _balance_chunks(self, chunks):
        """Balancerer chunks til optimal størrelse for søgning med dynamiske størrelser"""
        if not chunks:
            return []
            
        st.write("Balancerer chunk-størrelser...")
        
        # Find meget små chunks (under min_chunk_size)
        min_chunk_size = getattr(st.session_state, 'min_chunk_size', 250)
        
        small_chunks = []
        normal_chunks = []
        large_chunks = []
        
        for chunk in chunks:
            content_length = len(chunk["content"])
            chunk_type = chunk["metadata"].get("chunk_type", "text")
            # Brug den dynamiske målstørrelse for denne type
            target_size = self._get_target_size_for_chunk_type(chunk_type)
            
            if content_length < min_chunk_size:
                small_chunks.append(chunk)
            elif content_length > target_size * 1.5:
                large_chunks.append(chunk)
            else:
                normal_chunks.append(chunk)
        
        st.write(f"Fandt {len(small_chunks)} små chunks, {len(normal_chunks)} normale chunks, og {len(large_chunks)} for store chunks")
        
        # 1. Kombinér små chunks hvis de har samme section_id
        section_grouped_chunks = {}
        for chunk in small_chunks:
            key = (chunk["metadata"]["section"], chunk["metadata"].get("subsection", ""))
            if key not in section_grouped_chunks:
                section_grouped_chunks[key] = []
            section_grouped_chunks[key].append(chunk)
        
        merged_chunks = []
        for key, group in section_grouped_chunks.items():
            if len(group) <= 1:
                merged_chunks.extend(group)  # Behold enkeltstående små chunks
                continue
                
            # Sorter efter position, hvis den findes
            group.sort(key=lambda c: c["metadata"].get("chunk_position", 0))
            
            current_content = ""
            current_metadata = None
            chunk_type = group[0]["metadata"].get("chunk_type", "text")
            target_size = self._get_target_size_for_chunk_type(chunk_type)
            
            for chunk in group:
                # Start ny metadata hvis nødvendigt
                if not current_metadata:
                    current_metadata = chunk["metadata"].copy()
                
                # Hvis tilføjelse af denne chunk ville overskride målstørrelsen, gem nuværende
                if current_content and len(current_content + "\n\n" + chunk["content"]) > target_size:
                    merged_chunks.append({
                        "content": current_content,
                        "metadata": current_metadata
                    })
                    current_content = chunk["content"]
                    current_metadata = chunk["metadata"].copy()
                else:
                    # Tilføj chunk til nuværende
                    if current_content:
                        current_content += "\n\n"
                    current_content += chunk["content"]
                    
                    # Kombiner metadata (f.eks. lister af referencer)
                    for field in ["law_references", "case_references", "concepts", 
                                 "legal_exceptions", "affected_groups"]:
                        if field in chunk["metadata"] and field in current_metadata:
                            if isinstance(current_metadata[field], list) and isinstance(chunk["metadata"][field], list):
                                combined = set()
                                
                                # Håndtér både simple lister og lister af dictionaries
                                if field == "law_references" and current_metadata[field] and chunk["metadata"][field] and all(isinstance(item, dict) for item in current_metadata[field] + chunk["metadata"][field]):
                                    # For strukturerede lovhenvisninger, brug ref-værdierne til sammenligning
                                    ref_dict = {}
                                    for item in current_metadata[field] + chunk["metadata"][field]:
                                        ref = item["ref"]
                                        if ref not in ref_dict:
                                            ref_dict[ref] = item
                                        elif item.get("is_primary", False):
                                            # Behold is_primary flag hvis det er sat
                                            ref_dict[ref]["is_primary"] = True
                                    
                                    current_metadata[field] = list(ref_dict.values())
                                else:
                                    # For simple lister, fjern duplikater
                                    combined = set(current_metadata[field] + chunk["metadata"][field])
                                    current_metadata[field] = list(combined)
                    
                    # Håndter question_types særskilt, da det kan være et nyt felt
                    if "question_types" in chunk["metadata"]:
                        if "question_types" not in current_metadata:
                            current_metadata["question_types"] = chunk["metadata"]["question_types"]
                        else:
                            combined = set(current_metadata["question_types"] + chunk["metadata"]["question_types"])
                            current_metadata["question_types"] = list(combined)
            
            # Tilføj sidste kombinerede chunk
            if current_content:
                merged_chunks.append({
                    "content": current_content,
                    "metadata": current_metadata
                })
        
        # 2. Del store chunks op på sætningsgrænser
        divided_chunks = []
        
        for chunk in large_chunks:
            content = chunk["content"]
            metadata = chunk["metadata"].copy()
            chunk_type = metadata.get("chunk_type", "text")
            # Brug den dynamiske målstørrelse for denne type
            target_size = self._get_target_size_for_chunk_type(chunk_type)
            
            # Del indhold ved sætningsgrænser med respekt for dynamisk målstørrelse
            segments = self._split_by_size(content, target_size=target_size)
            
            # Opret nye chunks for hvert segment
            for i, segment in enumerate(segments):
                if not segment.strip():
                    continue
                    
                new_chunk = {
                    "content": segment,
                    "metadata": metadata.copy()
                }
                
                # Opdater chunk_id for at undgå kollisioner
                new_chunk["metadata"]["chunk_id"] = f"{metadata['chunk_id']}_{i}"
                
                divided_chunks.append(new_chunk)
        
        # Kombiner alle chunks
        balanced_chunks = normal_chunks + merged_chunks + divided_chunks
        
        # Opdater retrievability score for alle chunks
        for chunk in balanced_chunks:
            chunk_type = chunk["metadata"].get("chunk_type", "text")
            chunk["metadata"]["retrievability"] = self._calculate_retrievability_enhanced(
                chunk["content"], 
                chunk_type, 
                chunk["metadata"].get("law_references", []), 
                chunk["metadata"].get("case_references", []),
                chunk["metadata"].get("concepts", [])
            )
        
        st.write(f"Efter balancering: {len(balanced_chunks)} chunks")
        
        return balanced_chunks

    def _normalize_law_references(self, chunks):
        """Normaliserer lovhenvisninger til standardformat baseret på konfiguration"""
        for chunk in chunks:
            metadata = chunk["metadata"]
            if "law_references" in metadata:
                normalized_refs = []
                
                # Håndter både strukturerede og ustrukturerede referencer
                if isinstance(metadata["law_references"], list):
                    if all(isinstance(item, dict) for item in metadata["law_references"]):
                        # Strukturerede referencer
                        for ref_obj in metadata["law_references"]:
                            ref = ref_obj["ref"]
                            normalized = ref
                            
                            # Normaliser reference-teksten baseret på konfigurationen
                            for lovnavn, abbr in self.law_abbreviations.items():
                                if lovnavn.lower() in ref.lower():
                                    para_match = re.search(r'§\s*(\d+\s*[A-Za-z]?)', ref)
                                    stk_match = re.search(r'(?:stk\.|stykke)\s*(\d+)', ref)
                                    
                                    if para_match:
                                        normalized = f"{abbr} § {para_match.group(1).strip()}"
                                        if stk_match:
                                            normalized += f", stk. {stk_match.group(1)}"
                                        break
                            
                            # Tjek om det er en direkte paragrafhenvisning
                            if normalized == ref and ref.startswith("§"):
                                para_match = re.search(r'§\s*(\d+\s*[A-Za-z]?)', ref)
                                stk_match = re.search(r'(?:stk\.|stykke)\s*(\d+)', ref)
                                
                                if para_match:
                                    # Brug dynamisk bestemt lovforkortelse eller default
                                    lovprefix = self._determine_law_from_context(ref)
                                    normalized = f"{lovprefix} § {para_match.group(1).strip()}"
                                    if stk_match:
                                        normalized += f", stk. {stk_match.group(1)}"
                            
                            # Opret nyt reference-objekt med normaliseret tekst
                            normalized_refs.append({
                                "ref": normalized,
                                "is_primary": ref_obj.get("is_primary", False)
                            })
                    else:
                        # Ustrukturerede referencer
                        for ref in metadata["law_references"]:
                            normalized = ref
                            
                            for lovnavn, abbr in self.law_abbreviations.items():
                                if lovnavn.lower() in ref.lower():
                                    para_match = re.search(r'§\s*(\d+\s*[A-Za-z]?)', ref)
                                    stk_match = re.search(r'(?:stk\.|stykke)\s*(\d+)', ref)
                                    
                                    if para_match:
                                        normalized = f"{abbr} § {para_match.group(1).strip()}"
                                        if stk_match:
                                            normalized += f", stk. {stk_match.group(1)}"
                                        break
                            
                            if normalized == ref and ref.startswith("§"):
                                para_match = re.search(r'§\s*(\d+\s*[A-Za-z]?)', ref)
                                stk_match = re.search(r'(?:stk\.|stykke)\s*(\d+)', ref)
                                
                                if para_match:
                                    lovprefix = self._determine_law_from_context(ref)
                                    normalized = f"{lovprefix} § {para_match.group(1).strip()}"
                                    if stk_match:
                                        normalized += f", stk. {stk_match.group(1)}"
                            
                            normalized_refs.append(normalized)
                
                metadata["normalized_law_references"] = normalized_refs
        
        return chunks

    def _normalize_case_references(self, chunks):
        """Normaliserer domsreferencer til standardformat"""
        for chunk in chunks:
            metadata = chunk["metadata"]
            if "case_references" in metadata:
                normalized_refs = []
                
                for ref in metadata["case_references"]:
                    normalized = ref
                    
                    # Normalisér danske domsreferencer til standardformat
                    # Højesteretsdomme (U/UfR)
                    u_match = re.search(r'U[fF]?R?\s*(\d{4})[.,]\s*(\d+)(?:\s*([A-ZØ]+))?', ref)
                    if u_match:
                        if u_match.group(3):
                            normalized = f"U.{u_match.group(1)}.{u_match.group(2)}.{u_match.group(3)}"
                        else:
                            normalized = f"U.{u_match.group(1)}.{u_match.group(2)}"
                    
                    # Skattesager (SKM)
                    skm_match = re.search(r'SKM[-\s]*(\d{4})[.,]\s*(\d+)[.,]?\s*([A-Z]+)?', ref)
                    if skm_match:
                        if skm_match.group(3):
                            normalized = f"SKM.{skm_match.group(1)}.{skm_match.group(2)}.{skm_match.group(3)}"
                        else:
                            normalized = f"SKM.{skm_match.group(1)}.{skm_match.group(2)}"
                    
                    # Landsskatteretsafgørelser (LSR)
                    lsr_match = re.search(r'LSR\s*[-\s]*(\d{4})[.,]\s*(\d+)', ref)
                    if lsr_match:
                        normalized = f"LSR.{lsr_match.group(1)}.{lsr_match.group(2)}"
                    
                    # Tidsskrift for Skatter og Afgifter (TfS)
                    tfs_match = re.search(r'TfS\s*(\d{4})[.,]\s*(\d+)(?:\s*([A-ZØ]+))?', ref)
                    if tfs_match:
                        if tfs_match.group(3):
                            normalized = f"TfS.{tfs_match.group(1)}.{tfs_match.group(2)}.{tfs_match.group(3)}"
                        else:
                            normalized = f"TfS.{tfs_match.group(1)}.{tfs_match.group(2)}"
                    
                    normalized_refs.append(normalized)
                
                metadata["normalized_case_references"] = normalized_refs
        
        return chunks

    def _add_cross_references(self, chunks):
        """Tilføjer krydsreferencer mellem chunks med vægtede relationer"""
        # Opbyg indeks over chunks
        chunk_index = {}
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            
            # Indekser efter afsnit+underafsnit
            section_key = f"{metadata['section']}_{metadata.get('subsection', '')}"
            if section_key not in chunk_index:
                chunk_index[section_key] = []
            chunk_index[section_key].append(i)
            
            # Indekser efter eksempelnumre
            if metadata.get("is_example") and metadata.get("example_num"):
                example_key = f"example_{metadata['example_num']}"
                chunk_index[example_key] = [i]
            
            # Indekser efter domme
            for case_ref in metadata.get("case_references", []):
                case_key = f"case_{case_ref}"
                if case_key not in chunk_index:
                    chunk_index[case_key] = []
                chunk_index[case_key].append(i)
            
            # Indekser efter lovhenvisninger
            law_refs = metadata.get("law_references", [])
            if law_refs:
                if isinstance(law_refs[0], dict) if law_refs else False:
                    # Strukturerede referencer
                    for ref_obj in law_refs:
                        law_key = f"law_{ref_obj['ref']}"
                        if law_key not in chunk_index:
                            chunk_index[law_key] = []
                        chunk_index[law_key].append(i)
                else:
                    # Ustrukturerede referencer
                    for ref in law_refs:
                        law_key = f"law_{ref}"
                        if law_key not in chunk_index:
                            chunk_index[law_key] = []
                        chunk_index[law_key].append(i)
            
            # Indekser efter koncepter
            for concept in metadata.get("concepts", []):
                concept_key = f"concept_{concept.lower()}"
                if concept_key not in chunk_index:
                    chunk_index[concept_key] = []
                chunk_index[concept_key].append(i)
        
        # Tilføj krydsreferencer til hvert chunk
        for i, chunk in enumerate(chunks):
            metadata = chunk["metadata"]
            related_chunks = []
            relation_weights = {}  # For at vægte relationerne
            
            # Find relaterede afsnit baseret på hierarki
            if "hierarchy_path" in metadata:
                for parent in metadata["hierarchy_path"][:-1]:  # Alle undtagen selve afsnittet
                    parent_key = f"{parent}_"  # Match alle underafsnit
                    for key in chunk_index:
                        if key.startswith(parent_key):
                            related_chunks.extend(chunk_index[key])
                            # Giv hierarkiske relationer en moderat vægt
                            for rel_idx in chunk_index[key]:
                                relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + 3
            
            # Find relaterede eksempler
            if metadata.get("subsection") and not metadata.get("is_example"):
                # Dette er et regel-afsnit, find tilhørende eksempler
                section_key = metadata["section"]
                for key in chunk_index:
                    if key.startswith(f"{section_key}_") and "example_" in key:
                        related_chunks.extend(chunk_index[key])
                        # Giv eksempelrelationer en høj vægt
                        for rel_idx in chunk_index[key]:
                            relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + 5
            
            # Find relaterede domme
            for case_ref in metadata.get("case_references", []):
                case_key = f"case_{case_ref}"
                if case_key in chunk_index:
                    related_chunks.extend(chunk_index[case_key])
                    # Giv domsreferencer en høj vægt
                    for rel_idx in chunk_index[case_key]:
                        relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + 5
            
            # Find relaterede love
            law_refs = metadata.get("law_references", [])
            if law_refs:
                if isinstance(law_refs[0], dict) if law_refs else False:
                    # Strukturerede referencer
                    for ref_obj in law_refs:
                        law_key = f"law_{ref_obj['ref']}"
                        if law_key in chunk_index:
                            related_chunks.extend(chunk_index[law_key])
                            # Giv lovrelationer en høj vægt (ekstra vægt til primære referencer)
                            weight = 7 if ref_obj.get("is_primary", False) else 5
                            for rel_idx in chunk_index[law_key]:
                                relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + weight
                else:
                    # Ustrukturerede referencer
                    for ref in law_refs:
                        law_key = f"law_{ref}"
                        if law_key in chunk_index:
                            related_chunks.extend(chunk_index[law_key])
                            # Giv lovrelationer en høj vægt
                            for rel_idx in chunk_index[law_key]:
                                relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + 5
            
            # Find relaterede koncepter
            for concept in metadata.get("concepts", []):
                concept_key = f"concept_{concept.lower()}"
                if concept_key in chunk_index:
                    related_chunks.extend(chunk_index[concept_key])
                    # Giv konceptrelationer en moderat vægt
                    for rel_idx in chunk_index[concept_key]:
                        relation_weights[rel_idx] = relation_weights.get(rel_idx, 0) + 2
            
            # Fjern selvreferencer og dubletter
            related_chunks = [idx for idx in related_chunks if idx != i]
            related_chunks = list(set(related_chunks))
            
            # Sortér efter vægt (de mest relevante først)
            sorted_related = sorted(related_chunks, key=lambda idx: relation_weights.get(idx, 0), reverse=True)
            
            # Tilføj krydsreferencer til metadata
            if sorted_related:
                metadata["related_chunks"] = []
                for rel_idx in sorted_related[:5]:  # Begræns til top 5
                    rel_chunk = chunks[rel_idx]
                    # Beregn relationstype
                    relation_type = self._determine_relation_type(chunk, rel_chunk)
                    
                    metadata["related_chunks"].append({
                        "chunk_id": rel_chunk["metadata"].get("chunk_id", ""),
                        "section": rel_chunk["metadata"].get("section", ""),
                        "subsection": rel_chunk["metadata"].get("subsection", ""),
                        "is_example": rel_chunk["metadata"].get("is_example", False),
                        "relation_type": relation_type,
                        "weight": relation_weights.get(rel_idx, 1)
                    })
        
        return chunks
    
    def _determine_relation_type(self, chunk1, chunk2):
        """Bestemmer typen af relation mellem to chunks"""
        meta1 = chunk1["metadata"]
        meta2 = chunk2["metadata"]
        
        # Eksempel relation
        if not meta1.get("is_example", False) and meta2.get("is_example", False):
            return "has_example"
        
        if meta1.get("is_example", False) and not meta2.get("is_example", False):
            return "example_of"
        
        # Hierarkisk relation
        if meta1.get("section") == meta2.get("section") and meta1.get("subsection") != meta2.get("subsection"):
            return "same_section_different_subsection"
        
        # Lovreference relation
        law_refs1 = meta1.get("law_references", [])
        law_refs2 = meta2.get("law_references", [])
        
        if law_refs1 and law_refs2:
            # Håndtér både strukturerede og ustrukturerede referencer
            if isinstance(law_refs1[0], dict) if law_refs1 else False:
                # Strukturerede referencer
                refs1 = [ref_obj["ref"] for ref_obj in law_refs1]
                
                if isinstance(law_refs2[0], dict) if law_refs2 else False:
                    # Begge er strukturerede
                    refs2 = [ref_obj["ref"] for ref_obj in law_refs2]
                else:
                    # Anden er ustruktureret
                    refs2 = law_refs2
                    
                common_laws = set(refs1) & set(refs2)
                if common_laws:
                    # Tjek om der er fælles primære referencer
                    primary_refs1 = [ref_obj["ref"] for ref_obj in law_refs1 if ref_obj.get("is_primary", False)]
                    
                    if isinstance(law_refs2[0], dict) if law_refs2 else False:
                        primary_refs2 = [ref_obj["ref"] for ref_obj in law_refs2 if ref_obj.get("is_primary", False)]
                    else:
                        primary_refs2 = []
                    
                    common_primary = set(primary_refs1) & set(primary_refs2)
                    if common_primary:
                        return "common_primary_law"
                    
                    return "common_law_reference"
            else:
                # Ustrukturerede referencer
                if isinstance(law_refs2[0], dict) if law_refs2 else False:
                    # Anden er struktureret
                    refs2 = [ref_obj["ref"] for ref_obj in law_refs2]
                    common_laws = set(law_refs1) & set(refs2)
                else:
                    # Begge er ustrukturerede
                    common_laws = set(law_refs1) & set(law_refs2)
                
                if common_laws:
                    return "common_law_reference"
        
        # Domsreference relation
        common_cases = set(meta1.get("case_references", [])) & set(meta2.get("case_references", []))
        if common_cases:
            return "common_case_reference"
        
        # Konceptrelation
        common_concepts = set(meta1.get("concepts", [])) & set(meta2.get("concepts", []))
        if common_concepts:
            return "common_concept"
        
        # Default
        return "related"

    def _add_legal_status(self, chunks):
        """Tilføjer juridisk status til chunks"""
        for chunk in chunks:
            content = chunk["content"].lower()
            metadata = chunk["metadata"]
            
            # Bestemmelse af juridisk status
            if re.search(r'\b(?:ophævet|bortfaldet|udgået|ikke længere gældende)\b', content):
                metadata["legal_status"] = "ophævet"
            elif re.search(r'\b(?:midlertidig|tidsbegrænset|gælder indtil|ophører den)\b', content):
                metadata["legal_status"] = "midlertidig"
                
                # Forsøg at finde udløbsdato
                date_match = re.search(r'(?:indtil|til|ophører|udløber)\s+(?:den)?\s+(\d{1,2}\.?\s+\w+\s+\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', content)
                if date_match:
                    metadata["expiry_date"] = date_match.group(1)
            else:
                metadata["legal_status"] = "gældende"
                
            # Noter som er knyttet til specifik lovgivning kan have mere specifik status
            if metadata.get("chunk_type") == "note" and metadata.get("law_references", []):
                # Undersøg om noten refererer til ophævet lovgivning
                if any("ophævet" in str(ref).lower() for ref in metadata["law_references"]):
                    metadata["legal_status"] = "historisk"
        
        return chunks

    def _ensure_complete_metadata(self, chunks):
        """Sikrer at alle chunks har komplette metadata"""
        required_fields = [
            "doc_id", "doc_type", "version_date", "section", 
            "section_title", "chunk_type", "is_example", "concepts",
            "law_references", "case_references", "complexity", "authority",
            "affected_groups", "legal_exceptions", "question_types"
        ]
        
        list_fields = ["concepts", "law_references", "case_references", "affected_groups", 
                      "legal_exceptions", "question_types"]
        
        for chunk in chunks:
            # Sikre at metadata findes
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            
            # Tilføj manglende felter
            for field in required_fields:
                if field not in chunk["metadata"]:
                    if field in list_fields:
                        chunk["metadata"][field] = []
                    else:
                        chunk["metadata"][field] = ""
            
            # Tilføj retrievability score hvis den mangler
            if "retrievability" not in chunk["metadata"]:
                chunk_type = chunk["metadata"].get("chunk_type", "text")
                chunk["metadata"]["retrievability"] = self._calculate_retrievability_enhanced(
                    chunk["content"], 
                    chunk_type, 
                    chunk["metadata"].get("law_references", []), 
                    chunk["metadata"].get("case_references", []),
                    chunk["metadata"].get("concepts", [])
                )
            
            # Sikr at legal_status findes
            if "legal_status" not in chunk["metadata"]:
                chunk["metadata"]["legal_status"] = "gældende"
        
        return chunks