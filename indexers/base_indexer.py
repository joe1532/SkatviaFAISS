# indexers/base_indexer.py
import streamlit as st

class BaseIndexer:
    """Basis-klasse for alle indekserere med fælles funktionalitet"""
    
    def __init__(self):
        """Initialiser indeksereren med standardværdier"""
        self.name = "Basis-indekserer"
        self.description = "Basis-indeksererklasse - brug ikke direkte"
    
    def display_settings(self, st):
        """
        Viser indstillinger for denne indekserer i Streamlit UI
        
        Args:
            st: Streamlit-objekt
            
        Returns:
            str: Nøgle der identificerer den valgte dokumenttype
        """
        st.warning("Basis-indeksereren har ingen indstillinger. Brug en specialiseret indekserer i stedet.")
        return "generisk"
    
    def process_document(self, text, doc_id, options):
        """
        Processer et dokument og generer chunks samt kontekstopsummering.
        
        Args:
            text: Råtekst fra dokumentet
            doc_id: Unikt dokument-ID
            options: Indstillinger for processering (dict)
            
        Returns:
            tuple: (chunks, context_summary)
        """
        raise NotImplementedError("BaseIndexer.process_document() skal implementeres af underklasser")
    
    def get_context_prompt_template(self, doc_type_key):
        """
        Hent kontekstprompt skabelonen baseret på dokumenttype.
        
        Args:
            doc_type_key: Nøgle for dokumenttypen
            
        Returns:
            str: Kontekstprompt skabelon
        """
        raise NotImplementedError("BaseIndexer.get_context_prompt_template() skal implementeres af underklasser")
    
    def get_indexing_prompt_template(self, doc_type_key, context_summary, doc_id, section_number):
        """
        Hent indekseringsprompt skabelonen baseret på dokumenttype.
        
        Args:
            doc_type_key: Nøgle for dokumenttypen
            context_summary: JSON-opsummering af dokumentets kontekst
            doc_id: Unikt dokument-ID
            section_number: Sektionsnummer (bruges til at identificere aktuelt segment)
            
        Returns:
            str: Indekseringsprompt skabelon
        """
        raise NotImplementedError("BaseIndexer.get_indexing_prompt_template() skal implementeres af underklasser")