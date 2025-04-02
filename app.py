# app.py - Simpel version med modulær indekseringsarkitektur
import os
import io
import time
import json
import uuid
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import re
import faiss
from datetime import datetime

# Importér vores moduler
from utils import storage
from utils import api_utils
from utils import pdf_utils
from utils import text_analysis
from utils import indexing
from utils import validation
from utils.optimization import cached_call_gpt4o, process_segments_parallel, optimize_chunks
from indexers import get_available_indexers, get_indexer_class

# Konfiguration
st.set_page_config(page_title="Skatteretlig Indekseringsværktøj", layout="wide")

# Initialisering af session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'doc_id' not in st.session_state:
    st.session_state.doc_id = None
if 'context_summary' not in st.session_state:
    st.session_state.context_summary = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_dict' not in st.session_state:
    st.session_state.embedding_dict = {}
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = None
if 'original_text_sections' not in st.session_state:
    st.session_state.original_text_sections = {}
if 'preserved_content' not in st.session_state:
    st.session_state.preserved_content = {}
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}
if 'filtered_chunks' not in st.session_state:
    st.session_state.filtered_chunks = []

# Forsøg at initialisere OpenAI-klienten
try:
    client = api_utils.get_openai_client()
except ValueError as e:
    st.error(str(e))
    st.info("Indtast din OpenAI API-nøgle for at fortsætte:")
    api_key = st.text_input("OpenAI API-nøgle", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        client = api_utils.get_openai_client()

def get_advanced_options():
    """Henter avancerede indstillinger fra Streamlit UI"""
    model_for_context = st.selectbox(
        "Model til kontekstanalyse",
        ["gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="GPT-4o giver bedre resultater, men er langsommere og har lavere rate limits"
    )
    
    max_text_per_request = st.slider(
        "Maks. tekstlængde per API-kald (tegn)",
        min_value=5000,
        max_value=100000,
        value=30000,
        step=5000,
        help="Lavere værdi reducerer risikoen for rate limit fejl, men øger antallet af API-kald"
    )
    
    wait_time_between_calls = st.slider(
        "Ventetid mellem API-kald (sekunder)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Længere ventetid reducerer risikoen for rate limit fejl"
    )
    
    return {
        "model": model_for_context,
        "max_text_length": max_text_per_request,
        "wait_time": wait_time_between_calls
    }

def display_context_summary(context_summary):
    """Viser kontekstopsummering i et pænt format."""
    if not isinstance(context_summary, dict):
        st.warning("Kontekstopsummering er ikke i forventet format.")
        st.json(context_summary)
        return
    
    # Udtræk de vigtigste dele af kontekstopsummeringen til visning
    summary = context_summary.get("summary", {})
    
    # Dokumentoplysninger
    st.subheader("Dokumentinformation")
    doc_info_col1, doc_info_col2 = st.columns(2)
    with doc_info_col1:
        st.write(f"**Type:** {context_summary.get('document_type', 'Ikke angivet')}")
        st.write(f"**Dato:** {context_summary.get('version_date', 'Ikke angivet')}")
    with doc_info_col2:
        st.write(f"**ID:** {context_summary.get('document_id', 'Ikke angivet')}")
    
    # Hovedtemaer og struktur
    st.subheader("Temaer og struktur")
    themes_col1, themes_col2 = st.columns(2)
    with themes_col1:
        st.write("**Hovedtemaer:**")
        for theme in summary.get("main_themes", []):
            st.write(f"- {theme}")
    with themes_col2:
        st.write("**Tematisk hierarki:**")
        for main_theme, sub_themes in summary.get("theme_hierarchy", {}).items():
            st.write(f"- {main_theme}")
            for sub_theme in sub_themes:
                st.write(f"  - {sub_theme}")
    
    # Juridiske undtagelser
    if "legal_exceptions" in summary:
        st.subheader("Juridiske undtagelser og specialtilfælde")
        for exception in summary.get("legal_exceptions", []):
            if isinstance(exception, dict):
                st.write(f"- **Regel:** {exception.get('rule', '')}")
                st.write(f"  **Undtagelse:** {exception.get('exception', '')}")
                st.write(f"  **Kilde:** {exception.get('source', '')}")
            else:
                st.write(f"- {exception}")
            st.write("")
    
    # Nøgleparagraffer
    if "key_paragraphs" in summary:
        st.subheader("Nøgleparagraffer")
        paragraphs = summary.get("key_paragraphs", {})
        for para, desc in paragraphs.items():
            st.write(f"**{para}:** {desc}")
    
    # Nøglekoncepter
    st.subheader("Nøglekoncepter og synonymer")
    concepts_col1, concepts_col2 = st.columns(2)
    with concepts_col1:
        st.write("**Centrale begreber:**")
        for concept in summary.get("key_concepts", [])[:10]:  # Begræns til 10
            st.write(f"- {concept}")
    with concepts_col2:
        st.write("**Begrebssynonymer:**")
        synonyms = summary.get("concept_synonyms", {})
        for concept, syns in list(synonyms.items())[:5]:  # Begræns til 5
            st.write(f"- {concept}: {', '.join(syns)}")
    
    # Noter oversigt
    if "notes_overview" in summary and summary["notes_overview"]:
        st.subheader("Oversigt over noter")
        notes = summary.get("notes_overview", {})
        note_df = pd.DataFrame([
            {
                "Note": note_num, 
                "Reference": ', '.join(note_data.get("references", [])) if isinstance(note_data.get("references", []), list) else str(note_data.get("references", "")), 
                "Indhold": note_data.get("text", "")[:100] + "...",
                "Juridiske undtagelser": ', '.join(note_data.get("key_legal_exceptions", [])) if isinstance(note_data.get("key_legal_exceptions", []), list) else "",
                "Prioritet": note_data.get("priority", "medium")
            }
            for note_num, note_data in list(notes.items())[:10]  # Begræns til 10 noter
        ])
        st.dataframe(note_df)
        
        # Vis antallet af noter
        st.info(f"Dokumentet indeholder {len(notes)} noter")
    
    # Referencer 
    if "additional_references" in summary:
        st.subheader("Referencer")
        refs = summary.get("additional_references", {})
        refs_col1, refs_col2 = st.columns(2)
        with refs_col1:
            st.write("**Ændringslove:**")
            for law in refs.get("amending_laws", [])[:5]:  # Begræns til 5
                st.write(f"- {law}")
            
            st.write("**Administrative afgørelser:**")
            for ruling in refs.get("administrative_rulings", [])[:5]:  # Begræns til 5
                st.write(f"- {ruling}")
        with refs_col2:
            st.write("**Litteratur:**")
            for lit in refs.get("literature", [])[:5]:  # Begræns til 5
                st.write(f"- {lit}")
            
            st.write("**Vigtige afgørelser:**")
            for case in refs.get("significant_cases", [])[:5]:  # Begræns til 5
                st.write(f"- {case}")
    
    # Midlertidige bestemmelser
    if "temporary_provisions" in summary:
        st.subheader("Midlertidige bestemmelser")
        for temp in summary.get("temporary_provisions", []):
            st.write(f"- {temp}")

def display_chunks(chunks, filter_type=None, filter_text=None):
    """Viser chunks med forbedrede filtreringsmuligheder."""
    filtered_chunks = chunks
    
    # Anvend filter efter type
    if filter_type:
        if filter_type == "Kun lovtekst":
            filtered_chunks = [c for c in chunks if not c["metadata"].get("is_note", False)]
        elif filter_type == "Kun noter":
            filtered_chunks = [c for c in chunks if c["metadata"].get("is_note", False)]
        elif filter_type == "Med krydsreferencer":
            filtered_chunks = [c for c in chunks if 
                            (c["metadata"].get("fortolkningsbidrag") and c["metadata"]["fortolkningsbidrag"]) or
                            c["metadata"].get("note_reference")]
        elif filter_type == "Midlertidige bestemmelser":
            filtered_chunks = [c for c in chunks if c["metadata"].get("status") == "midlertidig"]
        elif filter_type == "Med juridiske undtagelser":
            filtered_chunks = [c for c in chunks if 
                            c["metadata"].get("legal_exceptions") and len(c["metadata"]["legal_exceptions"]) > 0]
        elif filter_type == "Berørte persongrupper":
            filtered_chunks = [c for c in chunks if 
                            c["metadata"].get("affected_groups") and len(c["metadata"]["affected_groups"]) > 0]
        elif filter_type == "Uden referencer":
            filtered_chunks = [c for c in chunks if 
                            not c["metadata"].get("is_note", False) and 
                            (not c["metadata"].get("fortolkningsbidrag") or len(c["metadata"].get("fortolkningsbidrag", [])) == 0)]
        elif filter_type == "Høj prioritet":
            filtered_chunks = [c for c in chunks if c["metadata"].get("priority") == "høj"]
        elif filter_type == "Komplekse bestemmelser":
            filtered_chunks = [c for c in chunks if c["metadata"].get("complexity") == "kompleks"]
    
    # Anvend tekstfilter
    if filter_text:
        filter_text = filter_text.lower().strip()
        filtered_chunks = [
            c for c in filtered_chunks if 
            (c["metadata"].get("paragraph", "").lower().find(filter_text) >= 0) or
            (c["metadata"].get("stykke", "").lower().find(filter_text) >= 0) or
            (str(c["metadata"].get("note_number", "")).lower().find(filter_text) >= 0) or
            (c["content"].lower().find(filter_text) >= 0) or
            any(concept.lower().find(filter_text) >= 0 for concept in c["metadata"].get("concepts", [])) or
            any(group.lower().find(filter_text) >= 0 for group in c["metadata"].get("affected_groups", []))
        ]
    
    # Gem filtrerede chunks i session state til eksportering
    st.session_state.filtered_chunks = filtered_chunks
    
    # Vis antal chunks
    st.info(f"Viser {len(filtered_chunks)} af {len(chunks)} chunks")
    
    # Skab en mere detaljeret tabel med chunks
    chunk_df = pd.DataFrame([
        {
            "Type": "Note" if chunk["metadata"].get("is_note", False) else "Lovtekst",
            "Tekst": chunk["content"][:100] + "...",
            "Paragraf": chunk["metadata"].get("paragraph", ""),
            "Stykke": chunk["metadata"].get("stykke", ""),
            "Status": chunk["metadata"].get("status", "gældende"),
            "Tema": chunk["metadata"].get("theme", ""),
            "Undertema": chunk["metadata"].get("subtheme", ""),
            "Nøgleord": ", ".join(chunk["metadata"].get("concepts", [])),
            "Persongrupper": ", ".join(chunk["metadata"].get("affected_groups", [])),
            "Krydsreferencer": ", ".join(str(x) for x in chunk["metadata"].get("fortolkningsbidrag", [])) if not chunk["metadata"].get("is_note", False) else 
                            (str(chunk["metadata"].get("note_reference", "")) if chunk["metadata"].get("note_reference") else ""),
            "Note-nr": chunk["metadata"].get("note_number", "") if chunk["metadata"].get("is_note", False) else "",
            "Prioritet": chunk["metadata"].get("priority", "medium"),
            "Kompleksitet": chunk["metadata"].get("complexity", "moderat") 
        }
        for chunk in filtered_chunks
    ])
    st.dataframe(chunk_df)
    
    return filtered_chunks

def provide_download_options(chunks, context_summary, doc_id, faiss_index=None, embedding_dict=None):
    """Giver forbedrede muligheder for at downloade data."""
    st.subheader("Eksporter data")
    
    # Download chunks som JSON
    chunks_json = json.dumps(chunks, ensure_ascii=False, indent=2)
    st.download_button(
        label="Download chunks som JSON",
        data=chunks_json,
        file_name=f"chunks_{doc_id}.json",
        mime="application/json"
    )
    
    # Download kontekstopsummering
    if context_summary:
        context_json = json.dumps(context_summary, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download dokumentoversigt som JSON",
            data=context_json,
            file_name=f"kontekst_{doc_id}.json",
            mime="application/json"
        )
    
    # Download FAISS indeks og embeddings hvis de findes
    if faiss_index is not None and embedding_dict is not None:
        st.subheader("Download søgeindeks")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gem indeks til tempfil og læs bytes
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".faiss") as f:
                    faiss.write_index(faiss_index, f.name)
                    temp_name = f.name  # Gem navnet uden for with-blokken
                
                # Læs filen i en separat blok
                with open(temp_name, "rb") as f_read:
                    index_bytes = f_read.read()
                
                # Forsøg at slette filen efter brug
                try:
                    os.unlink(temp_name)  # Slet tempfil
                except PermissionError:
                    # Hvis filen er låst, ignorér fejlen
                    # Den slettes automatisk ved programafslutning
                    pass
                    
                st.download_button(
                    label="Download FAISS indeks",
                    data=index_bytes,
                    file_name=f"index_{doc_id}.faiss",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Kunne ikke eksportere FAISS indeks: {e}")
        
        with col2:
            # Serialisér embeddings dictionary
            embedding_bytes = pickle.dumps(embedding_dict)
            
            st.download_button(
                label="Download embeddings",
                data=embedding_bytes,
                file_name=f"embeddings_{doc_id}.pkl",
                mime="application/octet-stream"
            )

def document_listing_page():
    """Viser liste over indekserede dokumenter og mulighed for at indlæse dem."""
    st.header("Indekserede dokumenter")
    
    docs_df = storage.get_documents_dataframe()
    if docs_df.empty:
        st.info("Ingen dokumenter er indekseret endnu. Upload et dokument for at starte.")
        return
    
    # Vis dokumentoversigt
    st.dataframe(docs_df[["doc_id", "title", "document_type", "saved_at", "chunks_count"]])
    
    # Vælg et dokument at indlæse
    selected_doc = st.selectbox("Vælg dokument at indlæse:", docs_df["doc_id"].tolist())
    
    if st.button("Indlæs valgt dokument"):
        with st.spinner(f"Indlæser dokument {selected_doc}..."):
            document_data = storage.load_complete_document(selected_doc)
            
            if document_data:
                st.session_state.doc_id = selected_doc
                st.session_state.chunks = document_data["chunks"]
                st.session_state.context_summary = document_data["metadata"]
                st.session_state.faiss_index = document_data["index"]
                st.session_state.embedding_dict = document_data["embeddings"]
                st.session_state.processing_stats = document_data.get("stats", {})
                
                st.success(f"Dokumentet '{selected_doc}' blev indlæst med {len(document_data['chunks'])} chunks")
                st.rerun()
            else:
                st.error(f"Kunne ikke indlæse dokument {selected_doc}")
   
    # Mulighed for at slette et dokument
    if st.checkbox("Vis slettemuligheder"):
        doc_to_delete = st.selectbox("Vælg dokument at slette:", docs_df["doc_id"].tolist(), key="delete_selectbox")
        
        if st.button("Slet valgt dokument", type="primary", help="Dette kan ikke fortrydes!"):
            if storage.delete_document(doc_to_delete):
                st.success(f"Dokumentet '{doc_to_delete}' blev slettet")
                # Fjern fra session state hvis det var det aktive dokument
                if st.session_state.doc_id == doc_to_delete:
                    st.session_state.doc_id = None
                    st.session_state.chunks = []
                    st.session_state.context_summary = None
                    st.session_state.faiss_index = None
                    st.session_state.embedding_dict = {}
                st.rerun()
            else:
                st.error(f"Kunne ikke slette dokument {doc_to_delete}")

# Hovedfunktion
def main():
    st.title("Skatteretlig Indekseringsværktøj")

    st.markdown("""
    Denne applikation kan indeksere skatteretlige dokumenter og eksportere data til brug i andre apps.

    ### Trin:
    1. Vælg dokumenttype og upload en skatteretlig tekst som PDF
    2. Konfigurer indekseringen baseret på dokumenttypen
    3. Vent på indeksering med specialiserede algoritmer for den valgte dokumenttype
    4. Download chunks eller hele indekset til senere brug
    """)
    
    # Sikr at datamapper eksisterer
    storage.ensure_directories()
    
    # Valg af side
    page = st.sidebar.radio("Vælg side:", ["Upload og Indeksér", "Vis Indekserede Dokumenter"])
    
    if page == "Upload og Indeksér":
        # Avancerede indstillinger
        with st.expander("Avancerede indstillinger", expanded=False):
            options = get_advanced_options()
        
        # Upload sektion
        with st.expander("Upload dokument", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload en PDF-fil med skatteretligt indhold",
                type=["pdf"]
            )
            
            if uploaded_file:
                # Indlæs PDF
                with st.spinner("Indlæser PDF..."):
                    text, pdf_stats = pdf_utils.extract_text_from_pdf(uploaded_file)
                    st.session_state.raw_text = text
                    st.session_state.processing_stats.update(pdf_stats)
                
                # Valg af dokumenttype med radioknapper
                available_indexers = get_available_indexers()
                indexer_options = list(available_indexers.keys())

                doc_type = st.radio(
                    "Vælg dokumenttype:",
                    indexer_options,
                    format_func=lambda x: available_indexers.get(x, {}).get("display_name", x.capitalize())
                )
                # Instantier den valgte indekserer
                indexer_class = get_indexer_class(doc_type)
                indexer = indexer_class()
                
                # Vis dokumenttype-specifikke indstillinger
                doc_type_key = indexer.display_settings(st)
                
                # Generelle indstillinger for alle dokumenttyper
                st.session_state.identify_temporary = st.checkbox("Identificer midlertidige bestemmelser", value=True)
                st.session_state.validate_output = st.checkbox("Validér output og rapporter mangler", value=True)
                st.session_state.identify_legal_exceptions = st.checkbox("Identificer juridiske undtagelser", value=True)
                
                document_title = st.text_input("Dokumenttitel:", value=f"Skatteretligt dokument - {doc_type}")
                
                # Estimer tokens
                estimated_tokens = api_utils.estimate_tokens(text)
                st.info(f"Estimeret dokumentstørrelse: ~{estimated_tokens} tokens")
                
                if st.button("Indekser dokument"):
                    # Generer dokument ID
                    doc_id = f"{doc_type}_{uuid.uuid4().hex[:8]}"
                    st.session_state.doc_id = doc_id
                    
                    # Reset statistik
                    st.session_state.processing_stats = {}
                    
                    # Konfigurer processeringsindstillinger
                    processing_options = {
                        **options,  # Inkluder avancerede indstillinger
                        "doc_type_key": doc_type_key,
                        "preserve_notes": True,
                        "identify_exceptions": st.session_state.identify_legal_exceptions,
                        "identify_temporary": st.session_state.identify_temporary,
                        "validate_output": st.session_state.validate_output
                    }
                    
                    # Processer dokumentet med den specialiserede indekserer
                    with st.spinner("Processerer dokumentet..."):
                        chunks, context_summary = indexer.process_document(text, doc_id, processing_options)
                        
                        if chunks and context_summary:
                            st.session_state.chunks = chunks
                            st.session_state.context_summary = context_summary
                            
                            # Tilføj metadata
                            if "title" not in context_summary:
                                context_summary["title"] = document_title
                            if "doc_id" not in context_summary:
                                context_summary["doc_id"] = doc_id
                        else:
                            st.error("Indeksering fejlede. Prøv igen med andre indstillinger.")
                            return
                    
                    # Generer embeddings og opret FAISS indeks
                    if chunks:
                        with st.spinner("Bygger søgeindeks..."):
                            index, embedding_dict = indexing.build_faiss_index(chunks)
                            st.session_state.faiss_index = index
                            st.session_state.embedding_dict = embedding_dict
                            
                            if index is not None and embedding_dict:
                                st.success(f"Dokumentet er indekseret med {len(chunks)} chunks")
                                
                                # Gem data lokalt
                                with st.spinner("Gemmer indekseret dokument lokalt..."):
                                    storage.save_complete_document(
                                        doc_id, 
                                        context_summary,
                                        chunks,
                                        index,
                                        embedding_dict,
                                        st.session_state.processing_stats
                                    )
                                    st.success("Dokumentet er gemt lokalt og kan nu tilgås fra reader.py")
                            else:
                                st.error("Kunne ikke bygge søgeindeks")
                    else:
                        st.error("Indeksering fejlede: Ingen chunks blev genereret")
        
        # Vis kontekstopsummering hvis tilgængelig
        if st.session_state.context_summary:
            with st.expander("Dokumentoversigt", expanded=True):
                display_context_summary(st.session_state.context_summary)
                
                # Vis også den komplette JSON for kontekstopsummeringen
                if st.checkbox("Vis fuld JSON for kontekstopsummering"):
                    st.json(st.session_state.context_summary)
        
        # Vis indekserede chunks hvis tilgængelige
        if st.session_state.chunks:
            with st.expander("Indekserede chunks", expanded=True):
                st.header("Indekserede chunks")
                # Tilføj en dropdown til at filtrere chunks
                chunk_filter = st.selectbox(
                    "Filtrer chunks efter type:",
                    ["Alle", "Kun lovtekst", "Kun noter", "Med krydsreferencer", "Midlertidige bestemmelser", 
                     "Med juridiske undtagelser", "Berørte persongrupper", "Uden referencer", "Høj prioritet", "Komplekse bestemmelser"]
                )
                
                # Filter for specifikke noter eller paragraffer
                specific_filter = st.text_input("Filtrer efter specifik paragraf, stykke, notenummer eller nøgleord (f.eks. '§ 33 A', '794' eller 'grænsegængere'):")
                
                # Vis filtrerede chunks
                filtered_chunks = display_chunks(
                    st.session_state.chunks, 
                    filter_type=chunk_filter if chunk_filter != "Alle" else None,
                    filter_text=specific_filter if specific_filter else None
                )
                
                # Download muligheder
                provide_download_options(
                    st.session_state.chunks,
                    st.session_state.context_summary,
                    st.session_state.doc_id,
                    st.session_state.faiss_index,
                    st.session_state.embedding_dict
                )
    
    elif page == "Vis Indekserede Dokumenter":
        document_listing_page()

if __name__ == "__main__":
    main()