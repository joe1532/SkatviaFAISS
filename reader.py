# reader.py - opdateret version
import streamlit as st
import json
import os
import numpy as np
import time
import re
import tempfile
import pickle

# Importér vores moduler
from utils import storage
from utils import api_utils
from utils import indexing

# Konfiguration
st.set_page_config(page_title="Skatteretlig JSON Spørgetjeneste", layout="wide")

# Initialisering af session state
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'query_results' not in st.session_state:
    st.session_state.query_results = []
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedding_dict' not in st.session_state:
    st.session_state.embedding_dict = None

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

# Forbedringer til reader.py

def build_legal_context(results, question):
    """
    Opbygger en sammenhængende juridisk kontekst baseret på søgeresultater.
    Forbedret til at inkludere struktureret data og relationelle forbindelser.
    """
    # Opdel chunks efter type
    law_chunks = [r for r in results if not r["chunk"].get("metadata", {}).get("is_note", False)]
    note_chunks = [r for r in results if r["chunk"].get("metadata", {}).get("is_note", False)]
    
    # Sortér paragraffer og stykker i logisk rækkefølge
    def safe_sort_key(result):
        metadata = result["chunk"].get("metadata", {})
        paragraph = str(metadata.get("paragraph", ""))
        stykke = str(metadata.get("stykke", ""))
        return (paragraph, stykke)
    
    law_chunks.sort(key=safe_sort_key)
    
    # Sorter noter efter notenummer
    note_chunks.sort(key=lambda r: str(r["chunk"].get("metadata", {}).get("note_number", "")))
    
    # Identificér persongrupper og specialregler på tværs af resultaterne
    affected_groups = set()
    legal_exceptions = set()
    
    for result in results:
        metadata = result["chunk"].get("metadata", {})
        affected_groups.update(metadata.get("affected_groups", []))
        for exception in metadata.get("legal_exceptions", []):
            if isinstance(exception, dict):
                legal_exceptions.add(exception.get("exception", ""))
            else:
                legal_exceptions.add(exception)
    
    # Opbyg konteksten med forbedret struktur
    context = ""
    
    # Oversigt over persongrupper og juridiske undtagelser
    if affected_groups or legal_exceptions:
        context += "\n\n--- RELEVANTE SPECIALREGLER OG MÅLGRUPPER ---\n\n"
        
        if affected_groups:
            context += "Særlige persongrupper: " + ", ".join(affected_groups) + "\n\n"
            
        if legal_exceptions:
            context += "Juridiske undtagelser/specialregler:\n"
            for exception in legal_exceptions:
                context += f"- {exception}\n"
        
        context += "\n"
    
    # Lovtekst-sektion
    if law_chunks:
        context += "\n\n--- LOVTEKST ---\n\n"
        for r in law_chunks:
            metadata = r["chunk"].get("metadata", {})
            paragraph = metadata.get("paragraph", "")
            stykke = metadata.get("stykke", "")
            status = metadata.get("status", "gældende")
            
            # Markér midlertidige bestemmelser
            status_marker = ""
            if status == "midlertidig":
                expiry_date = metadata.get("expiry_date", "")
                status_marker = f" [MIDLERTIDIG til {expiry_date}]"
            elif status == "ophævet":
                status_marker = " [OPHÆVET]"
            
            # Inkluder fortolkningsbidrag i headeren
            fortolkningsbidrag = metadata.get("fortolkningsbidrag", [])
            fortolkning_marker = ""
            if fortolkningsbidrag:
                fortolkning_marker = f" [Fortolket i noter: {', '.join(fortolkningsbidrag)}]"
            
            # Lav header for denne chunk
            header = f"[{paragraph}"
            if stykke:
                header += f", {stykke}"
            header += f"]{status_marker}{fortolkning_marker}:"
            
            context += f"{header}\n{r['chunk'].get('content', '')}\n\n"
    
    # Note-sektion
    if note_chunks:
        context += "\n\n--- NOTER OG FORTOLKNINGSBIDRAG ---\n\n"
        for r in note_chunks:
            metadata = r["chunk"].get("metadata", {})
            note_number = metadata.get("note_number", "")
            
            # Find hvilke paragraffer noten fortolker
            note_reference = metadata.get("note_reference", "")
            reference_certainty = metadata.get("reference_certainty", "")
            
            reference_str = f" (fortolker "
            if isinstance(note_reference, list):
                refs = []
                for ref in note_reference:
                    if isinstance(ref, dict):
                        ref_str = ref.get("paragraph", "")
                        if ref.get("stykke"):
                            ref_str += f", {ref.get('stykke')}"
                        refs.append(ref_str)
                    else:
                        refs.append(str(ref))
                reference_str += ", ".join(refs)
            else:
                reference_str += str(note_reference)
            
            if reference_certainty:
                reference_str += f", {reference_certainty} reference"
            reference_str += ")"
            
            priority = metadata.get("priority", "medium")
            priority_marker = ""
            if priority == "høj":
                priority_marker = " [VIGTIG]"
            
            context += f"[Note {note_number}]{priority_marker}{reference_str}:\n{r['chunk'].get('content', '')}\n\n"
    
    # Relevante domme og afgørelser
    relevant_cases = []
    for result in results:
        metadata = result["chunk"].get("metadata", {})
        normalized_refs = metadata.get("normalized_references", [])
        for ref in normalized_refs:
            if ref.startswith(("SKM.", "TfS.", "U.")):
                relevant_cases.append(ref)
    
    if relevant_cases:
        relevant_cases = list(set(relevant_cases))  # Fjern dubletter
        context += "\n\n--- RELEVANTE DOMME OG AFGØRELSER ---\n\n"
        for case in relevant_cases:
            context += f"- {case}\n"
    
    return context

def create_legal_prompt(question, context):
    """
    Skaber en juridisk prompt baseret på spørgsmål og kontekst.
    """
    prompt = f"""
    Du er en ekspert i dansk skatteret. Baseret på følgende uddrag fra skatteretlige dokumenter, besvar dette spørgsmål så præcist som muligt:
    
    Spørgsmål: {question}
    
    Dokumentuddrag:
    {context}
    
    Analyser uddraget og giv et svar, der tager højde for alle relevante betingelser og regler. Henvis specifikt til paragraffer og stykker hvor relevant. Hvis uddraget ikke indeholder tilstrækkelig information, angiv hvad der mangler. Inddrag noter og fortolkningsbidrag, hvis de er relevante.
    """
    
    return prompt

def load_document_page():
    """
    Side til at indlæse tidligere indekserede dokumenter.
    """
    st.header("Indlæs indekserede dokumenter")
    
    # Hent liste over indekserede dokumenter
    docs_df = storage.get_documents_dataframe()
    if docs_df.empty:
        st.info("Ingen dokumenter fundet. Brug app.py til at indeksere et dokument først.")
        return False
    
    # Vis liste over dokumenter
    st.dataframe(docs_df[["title", "document_type", "saved_at", "chunks_count"]])
    
    # Vis admin muligheder
    show_admin = st.checkbox("Vis administrationsmuligheder", key="show_admin_options")
    
    if show_admin:
        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["Indlæs dokumenter", "Omdøb dokumenter", "Slet dokumenter"])
        
        with admin_tab1:
            # Multi-select dokument
            selected_docs = st.multiselect(
                "Vælg dokumenter at indlæse:", 
                docs_df["doc_id"].tolist(),
                default=docs_df["doc_id"].tolist(),  # Vælg alle som standard
                format_func=lambda x: f"{docs_df[docs_df['doc_id']==x]['title'].values[0]}"
            )
            
            if st.button("Indlæs valgte dokumenter"):
                if not selected_docs:
                    st.warning("Ingen dokumenter valgt.")
                    return False
                
                total_chunks = 0
                loaded_docs = []
                all_chunks = []
                
                # Opret en progress bar
                progress = st.progress(0.0)
                status_text = st.empty()
                
                for i, doc_id in enumerate(selected_docs):
                    status_text.text(f"Indlæser dokument {i+1}/{len(selected_docs)}: {doc_id}")
                    with st.spinner(f"Indlæser dokument {doc_id}..."):
                        document_data = storage.load_complete_document(doc_id)
                        
                        if document_data:
                            # Tilføj chunks til samlet liste
                            all_chunks.extend(document_data["chunks"])
                            total_chunks += len(document_data["chunks"])
                            loaded_docs.append(doc_id)
                            
                            # Opdater embeddings dictionary hvis vi bruger det første dokuments index
                            if i == 0:
                                st.session_state.faiss_index = document_data["index"]
                                st.session_state.embedding_dict = document_data["embeddings"]
                                st.session_state.metadata = document_data["metadata"]
                        else:
                            st.error(f"Kunne ikke indlæse dokument {doc_id}")
                    
                    # Opdater progress bar
                    progress.progress((i + 1) / len(selected_docs))
                
                if loaded_docs:
                    st.session_state.chunks = all_chunks
                    
                    st.success(f"Indlæst {len(loaded_docs)} dokumenter med i alt {total_chunks} chunks")
                    return True
                else:
                    st.error("Kunne ikke indlæse nogen dokumenter")
                    return False
        
        with admin_tab2:
            # Omdøbning af dokumenter
            doc_to_rename = st.selectbox(
                "Vælg dokument at omdøbe:", 
                docs_df["doc_id"].tolist(),
                format_func=lambda x: f"{docs_df[docs_df['doc_id']==x]['title'].values[0]}"
            )
            
            if doc_to_rename:
                current_title = docs_df[docs_df['doc_id']==doc_to_rename]['title'].values[0]
                new_doc_id = st.text_input("Nyt dokument-ID:", value=doc_to_rename)
                new_title = st.text_input("Ny titel:", value=current_title)
                
                if st.button("Omdøb dokument"):
                    if new_doc_id == doc_to_rename and new_title == current_title:
                        st.warning("Ingen ændringer at gemme.")
                    elif storage.document_exists(new_doc_id) and new_doc_id != doc_to_rename:
                        st.error(f"Et dokument med ID '{new_doc_id}' findes allerede.")
                    else:
                        with st.spinner("Omdøber dokument..."):
                            if storage.rename_document(doc_to_rename, new_doc_id, new_title):
                                st.success(f"Dokumentet blev omdøbt til '{new_doc_id}' med titlen '{new_title}'")
                                st.rerun()  # Opdater siden for at reflektere ændringen
                            else:
                                st.error("Kunne ikke omdøbe dokumentet. Prøv igen.")
        
        with admin_tab3:
            # Sletning af dokumenter
            doc_to_delete = st.selectbox(
                "Vælg dokument at slette:", 
                docs_df["doc_id"].tolist(),
                key="delete_selectbox",
                format_func=lambda x: f"{docs_df[docs_df['doc_id']==x]['title'].values[0]}"
            )
            
            if doc_to_delete:
                st.warning(f"Du er ved at slette dokumentet '{doc_to_delete}'.")
                confirm_delete = st.checkbox("Jeg forstår, at dette vil slette dokumentet permanent og ikke kan fortrydes.")
                
                if confirm_delete and st.button("Slet dokument", type="primary"):
                    with st.spinner("Sletter dokument..."):
                        if storage.delete_document(doc_to_delete):
                            st.success(f"Dokumentet '{doc_to_delete}' blev slettet")
                            st.rerun()  # Opdater siden for at reflektere ændringen
                        else:
                            st.error(f"Kunne ikke slette dokument {doc_to_delete}")
    else:
        # Behold den oprindelige funktionalitet for enkelthedens skyld
        selected_doc = st.selectbox("Vælg dokument at indlæse:", docs_df["doc_id"].tolist(),
                                    format_func=lambda x: f"{docs_df[docs_df['doc_id']==x]['title'].values[0]}")
        
        if st.button("Indlæs valgt dokument"):
            with st.spinner(f"Indlæser dokument {selected_doc}..."):
                document_data = storage.load_complete_document(selected_doc)
                
                if document_data:
                    st.session_state.chunks = document_data["chunks"]
                    st.session_state.metadata = document_data["metadata"]
                    st.session_state.faiss_index = document_data["index"]
                    st.session_state.embedding_dict = document_data["embeddings"]
                    
                    st.success(f"Dokumentet '{document_data['metadata'].get('title', selected_doc)}' er indlæst med {len(document_data['chunks'])} chunks")
                    
                    # Vis dokumentinfo
                    st.subheader("Dokumentinformation")
                    st.write(f"**Titel:** {document_data['metadata'].get('title', 'Ukendt')}")
                    st.write(f"**Type:** {document_data['metadata'].get('document_type', 'Ukendt')}")
                    st.write(f"**Dato:** {document_data['metadata'].get('version_date', 'Ukendt')}")
                    
                    return True
                else:
                    st.error(f"Kunne ikke indlæse dokument {selected_doc}")
                    return False
    
    return False

def upload_json_page():
    """
    Side til at uploade JSON-fil med indekserede chunks.
    """
    st.header("Upload JSON-fil")
    
    uploaded_file = st.file_uploader("Upload JSON-fil med indekserede chunks", type=["json"])
    
    if uploaded_file and st.button("Indlæs JSON"):
        try:
            content = uploaded_file.read().decode("utf-8")
            json_data = json.loads(content)
            if isinstance(json_data, dict) and "chunks" in json_data:
                st.session_state.chunks = json_data["chunks"]
            elif isinstance(json_data, list):
                st.session_state.chunks = json_data
            else:
                st.error("Uventet JSON-format. Filen skal indeholde en liste af chunks eller et dict med en 'chunks' nøgle.")
                st.session_state.chunks = []
                return False
            
            if st.session_state.chunks:
                st.success(f"JSON-fil indlæst med {len(st.session_state.chunks)} chunks!")
                return True
        except Exception as e:
            st.error(f"Fejl ved indlæsning af JSON-fil: {e}")
            st.session_state.chunks = []
            return False
    
    return False

def main():
    st.title("Skatteretlig JSON Spørgetjeneste")

    st.markdown("""
    ## Hvordan fungerer det?
    1. Vælg et indekseret dokument eller upload en JSON-fil med indekserede chunks
    2. Stil et spørgsmål om det indekserede dokument
    3. Få et svar baseret på relevante dele af dokumentet
    """)
    
    # Sikr at datamapper eksisterer
    storage.ensure_directories()
    
    # Valg af indlæsningsmetode
    load_method = st.radio(
        "Vælg indlæsningsmetode:",
        ["Indlæs lokalt indekseret dokument", "Upload JSON-fil"]
    )
    
    document_loaded = False
    
    if load_method == "Indlæs lokalt indekseret dokument":
        document_loaded = load_document_page()
    else:
        document_loaded = upload_json_page()
    
    # Kun vis søge- og spørgefunktionalitet hvis et dokument er indlæst
    if document_loaded or st.session_state.chunks:
        # Byg søgeindeks hvis nødvendigt
        if (st.session_state.faiss_index is None or st.session_state.embedding_dict is None) and st.session_state.chunks:
            if st.button("Byg søgeindeks"):
                with st.spinner("Bygger søgeindeks fra chunks..."):
                    index, embedding_dict = indexing.build_faiss_index(st.session_state.chunks)
                    st.session_state.faiss_index = index
                    st.session_state.embedding_dict = embedding_dict
                    if index is not None and embedding_dict:
                        st.success("Søgeindeks bygget!")
                    else:
                        st.error("Kunne ikke bygge søgeindeks")
        
        # Vis chunks-oversigt
        with st.expander("Indekserede chunks", expanded=False):
            st.info(f"Indlæst {len(st.session_state.chunks)} chunks")
            if st.checkbox("Vis detaljer om chunks"):
                chunk_overview = []
                for i, chunk in enumerate(st.session_state.chunks[:50]):
                    overview = {
                        "Index": i,
                        "Indhold": chunk.get("content", "")[:100] + "..." if len(chunk.get("content", "")) > 100 else chunk.get("content", ""),
                        "Type": "Note" if chunk.get("metadata", {}).get("is_note", False) else "Lovtekst",
                        "Paragraf": chunk.get("metadata", {}).get("paragraph", ""),
                        "Stykke": chunk.get("metadata", {}).get("stykke", ""),
                        "Tema": chunk.get("metadata", {}).get("theme", "")
                    }
                    chunk_overview.append(overview)
                st.dataframe(chunk_overview)
                if len(st.session_state.chunks) > 50:
                    st.info("Viser kun de første 50 chunks af hensyn til performance.")
        
        # Spørgsmål-svar interface
        if st.session_state.faiss_index is not None and st.session_state.embedding_dict is not None:
            st.subheader("Stil et spørgsmål")
            
            question = st.text_input("Dit spørgsmål:", value=st.session_state.question)
            
            col1, col2 = st.columns(2)
            with col1:
                top_k = st.slider("Antal chunks at hente", min_value=1, max_value=20, value=10)
            with col2:
                model = st.selectbox("Vælg model", ["gpt-4o", "o1-mini"], index=0)
            
            if st.button("Søg og besvar") and question:
                st.session_state.question = question
                
                with st.spinner("Søger efter relevante dele af dokumentet..."):
                    search_results = indexing.advanced_semantic_search(
                        question, 
                        st.session_state.chunks,
                        st.session_state.faiss_index, 
                        st.session_state.embedding_dict,
                        top_k=top_k
                    )
                    st.session_state.query_results = search_results
                
                if search_results:
                    with st.spinner(f"Genererer svar med {model}..."):
                        context = build_legal_context(search_results, question)
                        prompt = create_legal_prompt(question, context)
                        response = api_utils.call_gpt4o(prompt, model=model, json_mode=False)
                        st.session_state.answer = response
                else:
                    st.warning("Ingen relevante dele af dokumentet fundet. Prøv at omformulere spørgsmålet.")
                    st.session_state.answer = ""
            
            if st.session_state.query_results:
                st.subheader("Relevante dele af dokumentet:")
                
                # Vis en oversigt over hvilke chunks der blev fundet
                law_chunks = [r for r in st.session_state.query_results if not r["chunk"].get("metadata", {}).get("is_note", False)]
                note_chunks = [r for r in st.session_state.query_results if r["chunk"].get("metadata", {}).get("is_note", False)]
                
                st.write(f"Fundet {len(law_chunks)} lovtekst-chunks og {len(note_chunks)} note-chunks:")
                st.write("Lovtekst:", [f"{r['chunk'].get('metadata', {}).get('paragraph', '')}-{r['chunk'].get('metadata', {}).get('stykke', '')}" for r in law_chunks])
                st.write("Noter:", [f"Note {r['chunk'].get('metadata', {}).get('note_number', '')}" for r in note_chunks])
                
                results_tab, answer_tab = st.tabs(["Søgeresultater", "Svar"])
                
                with results_tab:
                    # Vis lovtekst chunks
                    st.subheader("Lovtekst")
                    for i, result in enumerate(law_chunks):
                        chunk = result["chunk"]
                        score = result["score"]
                        metadata = chunk.get("metadata", {})
                        
                        title = f"{metadata.get('paragraph', '')}-{metadata.get('stykke', '')}"
                        
                        with st.expander(f"{title} (Score: {score:.4f})"):
                            st.markdown(f"**Indhold:**\n{chunk.get('content', '')}")
                            if st.checkbox(f"Vis metadata for {title}", key=f"metadata_law_{i}"):
                                st.json(metadata)
                    
                    # Vis note chunks
                    st.subheader("Noter og fortolkningsbidrag")
                    for i, result in enumerate(note_chunks):
                        chunk = result["chunk"]
                        score = result["score"]
                        metadata = chunk.get("metadata", {})
                        
                        title = f"Note {metadata.get('note_number', '')}"
                        
                        with st.expander(f"{title} (Score: {score:.4f})"):
                            st.markdown(f"**Indhold:**\n{chunk.get('content', '')}")
                            if st.checkbox(f"Vis metadata for {title}", key=f"metadata_note_{i}"):
                                st.json(metadata)
                
                with answer_tab:
                    if st.session_state.answer:
                        st.markdown(st.session_state.answer)
                    else:
                        st.info("Intet svar genereret endnu. Klik på 'Søg og besvar' for at få et svar.")

if __name__ == "__main__":
    main()