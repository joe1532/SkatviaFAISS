import os
import json
import pickle
import faiss
import shutil
import glob
import pandas as pd
from datetime import datetime

# Konstanter
DATA_DIR = "data"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

def ensure_directories():
    """Sikrer at de nødvendige mapper eksisterer."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)

def get_document_dir(doc_id):
    """Returnerer stien til mappen for et specifikt dokument."""
    return os.path.join(DOCUMENTS_DIR, doc_id)

def document_exists(doc_id):
    """Tjekker om et dokument med det angivne ID allerede eksisterer."""
    return os.path.exists(get_document_dir(doc_id))

def save_document_metadata(doc_id, metadata):
    """Gemmer dokumentets metadata."""
    ensure_directories()
    doc_dir = get_document_dir(doc_id)
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    # Tilføj tidsstempel for gemning
    metadata["saved_at"] = datetime.now().isoformat()
    
    with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def save_chunks(doc_id, chunks):
    """Gemmer chunks til et dokument."""
    ensure_directories()
    doc_dir = get_document_dir(doc_id)
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def save_faiss_index(doc_id, index, embedding_dict):
    """Gemmer FAISS-indeks og embeddings."""
    ensure_directories()
    doc_dir = get_document_dir(doc_id)
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    # Gem FAISS-indeks
    faiss.write_index(index, os.path.join(doc_dir, "index.faiss"))
    
    # Gem embeddings dictionary
    with open(os.path.join(doc_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(embedding_dict, f)

def save_processing_stats(doc_id, stats):
    """Gemmer processeringsstatistik."""
    ensure_directories()
    doc_dir = get_document_dir(doc_id)
    
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    
    with open(os.path.join(doc_dir, "stats.json"), "w", encoding="utf-8") as f:
        # Sikr at datoer er konverteret til strenge
        stats_serializable = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                stats_serializable[key] = {k: str(v) if isinstance(v, datetime) else v for k, v in value.items()}
            else:
                stats_serializable[key] = str(value) if isinstance(value, datetime) else value
        
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)

def load_document_metadata(doc_id):
    """Indlæser dokumentets metadata."""
    doc_dir = get_document_dir(doc_id)
    metadata_path = os.path.join(doc_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_chunks(doc_id):
    """Indlæser chunks fra et dokument."""
    doc_dir = get_document_dir(doc_id)
    chunks_path = os.path.join(doc_dir, "chunks.json")
    
    if not os.path.exists(chunks_path):
        return None
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_faiss_index(doc_id):
    """Indlæser FAISS-indeks."""
    doc_dir = get_document_dir(doc_id)
    index_path = os.path.join(doc_dir, "index.faiss")
    
    if not os.path.exists(index_path):
        return None
    
    return faiss.read_index(index_path)

def load_embeddings(doc_id):
    """Indlæser embeddings dictionary."""
    doc_dir = get_document_dir(doc_id)
    embeddings_path = os.path.join(doc_dir, "embeddings.pkl")
    
    if not os.path.exists(embeddings_path):
        return None
    
    with open(embeddings_path, "rb") as f:
        return pickle.load(f)

def load_processing_stats(doc_id):
    """Indlæser processeringsstatistik."""
    doc_dir = get_document_dir(doc_id)
    stats_path = os.path.join(doc_dir, "stats.json")
    
    if not os.path.exists(stats_path):
        return None
    
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)

def delete_document(doc_id):
    """Sletter et dokument og alle dets filer."""
    doc_dir = get_document_dir(doc_id)
    
    if os.path.exists(doc_dir):
        shutil.rmtree(doc_dir)
        return True
    
    return False

def list_documents():
    """Lister alle indekserede dokumenter."""
    ensure_directories()
    
    documents = []
    
    for doc_dir in glob.glob(os.path.join(DOCUMENTS_DIR, "*")):
        if os.path.isdir(doc_dir):
            doc_id = os.path.basename(doc_dir)
            metadata_path = os.path.join(doc_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Opret en enkel oversigt
                doc_info = {
                    "doc_id": doc_id,
                    "title": metadata.get("title", "Ukendt titel"),
                    "document_type": metadata.get("document_type", "Ukendt type"),
                    "version_date": metadata.get("version_date", "Ukendt dato"),
                    "saved_at": metadata.get("saved_at", "Ukendt gemmetidspunkt"),
                    "has_index": os.path.exists(os.path.join(doc_dir, "index.faiss")),
                    "chunks_count": len(json.load(open(os.path.join(doc_dir, "chunks.json"), "r", encoding="utf-8"))) if os.path.exists(os.path.join(doc_dir, "chunks.json")) else 0
                }
                documents.append(doc_info)
    
    # Sorter efter gemmetidspunkt (nyeste først)
    documents.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    
    return documents

def get_documents_dataframe():
    """Returnerer en dataframe med alle indekserede dokumenter."""
    docs = list_documents()
    if not docs:
        return pd.DataFrame()
    
    return pd.DataFrame(docs)

def save_complete_document(doc_id, metadata, chunks, index, embedding_dict, stats=None):
    """Gemmer alle data for et dokument i én funktion."""
    ensure_directories()
    
    save_document_metadata(doc_id, metadata)
    save_chunks(doc_id, chunks)
    save_faiss_index(doc_id, index, embedding_dict)
    
    if stats:
        save_processing_stats(doc_id, stats)
    
    return True

def load_complete_document(doc_id):
    """Indlæser alle data for et dokument i én funktion."""
    if not document_exists(doc_id):
        return None
    
    metadata = load_document_metadata(doc_id)
    chunks = load_chunks(doc_id)
    index = load_faiss_index(doc_id)
    embeddings = load_embeddings(doc_id)
    stats = load_processing_stats(doc_id)
    
    return {
        "metadata": metadata,
        "chunks": chunks,
        "index": index,
        "embeddings": embeddings,
        "stats": stats
    }
# I storage.py - tilføj denne funktion
def rename_document(old_doc_id, new_doc_id, new_title=None):
    """
    Omdøber et dokument ved at kopiere alle filer og opdatere metadata.
    
    Args:
        old_doc_id: Det gamle dokument-ID
        new_doc_id: Det nye dokument-ID
        new_title: Nyt dokumenttitel (valgfrit)
        
    Returns:
        Boolean der indikerer om omdøbningen lykkedes
    """
    if not document_exists(old_doc_id):
        return False
    
    if document_exists(new_doc_id):
        return False  # Kan ikke omdøbe til et ID der allerede eksisterer
    
    try:
        # Indlæs al data fra det gamle dokument
        old_data = load_complete_document(old_doc_id)
        if not old_data:
            return False
        
        # Opdater metadata med nyt ID og evt. ny titel
        old_data['metadata']['doc_id'] = new_doc_id
        if new_title:
            old_data['metadata']['title'] = new_title
        
        # Opdater doc_id i alle chunks
        for chunk in old_data['chunks']:
            chunk['metadata']['doc_id'] = new_doc_id
        
        # Gem alt under det nye ID
        save_complete_document(
            new_doc_id,
            old_data['metadata'],
            old_data['chunks'],
            old_data['index'],
            old_data['embeddings'],
            old_data.get('stats', {})
        )
        
        # Slet det gamle dokument
        delete_document(old_doc_id)
        
        return True
    except Exception as e:
        print(f"Fejl ved omdøbning af dokument: {e}")
        return False