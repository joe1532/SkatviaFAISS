import numpy as np
import streamlit as st
import faiss
import time
import re
from . import api_utils

def build_faiss_index(chunks, batch_size=20):
    """
    Bygger et FAISS-indeks fra chunks med batch-behandling af embeddings.
    
    Args:
        chunks: Liste af chunks
        batch_size: Antal chunks at behandle ad gangen
    
    Returns:
        FAISS-indeks og embedding dictionary
    """
    if not chunks:
        return None, {}
        
    with st.spinner("Genererer embeddings..."):
        embedding_dict = {}
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        
        # Behandl embeddings i batches for at reducere API-kald
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]
            
            for j, chunk in enumerate(batch):
                embedding = api_utils.generate_embedding(chunk["content"])
                if embedding:
                    embedding_dict[i + j] = {"embedding": embedding, "chunk": chunk}
            
            # Opdater fremskridt
            progress_bar.progress((end_idx) / total_chunks)
            
            # Vent mellem batches, ikke mellem hvert embedding
            if end_idx < total_chunks:
                time.sleep(1)
    
    # Resten af koden forbliver uændret...
    
    with st.spinner("Bygger FAISS indeks..."):
        if not embedding_dict:
            st.error("Ingen embeddings genereret!")
            return None, {}
        
        embedding_dim = len(list(embedding_dict.values())[0]["embedding"])  # 3072 for text-embedding-3-large
        num_chunks = len(embedding_dict)
        
        # Sæt nlist = 100 for ~10.000 chunks, ellers √n
        nlist = 100 if 5000 <= num_chunks <= 15000 else int(np.sqrt(num_chunks))
        if nlist < 1:
            nlist = 1
        
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        
        vectors = np.array([data["embedding"] for data in embedding_dict.values()], dtype=np.float32)
        
        if num_chunks < nlist:
            st.warning(f"For få chunks ({num_chunks}) til IVF. Bruger IndexFlatL2.")
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(vectors)
        else:
            index.train(vectors)
            index.add(vectors)
        
        return index, embedding_dict

def search_faiss_index(query, index, embedding_dict, top_k=10):
    """
    Søger i FAISS-indeks baseret på en forespørgsel.
    
    Args:
        query: Søgetekst
        index: FAISS-indeks
        embedding_dict: Dictionary med embeddings
        top_k: Antal resultater der returneres
    
    Returns:
        Liste af matchende chunks og deres scores
    """
    # Generer embedding for søgningen
    query_embedding = api_utils.generate_embedding(query)
    if not query_embedding:
        return []
    
    # Søg i FAISS index
    query_vector = np.array([query_embedding]).astype('float32')
    
    # Sæt antal clusters at søge i (nprobe)
    if hasattr(index, 'nprobe'):
        index.nprobe = min(10, index.ntotal)  # Søg i op til 10 clusters
    
    distances, indices = index.search(query_vector, top_k)
    
    # Konverter resultater til et format vi kan bruge
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(embedding_dict):
            continue
        chunk = embedding_dict[idx]["chunk"]
        results.append({
            "chunk": chunk,
            "score": float(1.0 / (1.0 + distances[0][i]))  # Konverter distance til score
        })
    
    return results

def advanced_semantic_search(query, chunks, index, embedding_dict, top_k=10):
    """
    Avanceret semantisk søgning der kombinerer FAISS med metadata-filtrering.
    
    Args:
        query: Søgetekst
        chunks: Liste af chunks
        index: FAISS-indeks
        embedding_dict: Dictionary med embeddings
        top_k: Antal resultater der returneres
    
    Returns:
        Liste af matchende chunks og deres scores
    """
    # 1. Identificer juridiske koncepter i forespørgslen
    concepts = identify_legal_concepts(query)
    
    # 2. Metadata-baseret filtrering
    metadata_results = filter_chunks_by_metadata(query, chunks, concepts)
    
    # 3. Standard semantisk søgning
    semantic_results = search_faiss_index(query, index, embedding_dict, top_k=top_k)
    
    # 4. Find paragraffer der er relevante i resultaterne
    relevant_paragraphs = []
    
    # Fra metadata-resultater
    for result in metadata_results:
        metadata = result["chunk"].get("metadata", {})
        if not metadata.get("is_note", False):  # Kun lovtekst, ikke noter
            paragraph = metadata.get("paragraph", "")
            stykke = metadata.get("stykke", "")
            if paragraph:
                para_key = f"{paragraph}"
                if stykke:
                    para_key += f" {stykke}"
                relevant_paragraphs.append(para_key)
    
    # Fra semantiske resultater
    for result in semantic_results:
        metadata = result["chunk"].get("metadata", {})
        if not metadata.get("is_note", False):  # Kun lovtekst, ikke noter
            paragraph = metadata.get("paragraph", "")
            stykke = metadata.get("stykke", "")
            if paragraph:
                para_key = f"{paragraph}"
                if stykke:
                    para_key += f" {stykke}"
                relevant_paragraphs.append(para_key)
    
    # 5. Find noter der relaterer sig til disse paragraffer
    related_notes = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if metadata.get("is_note", False):  # Kun noter
            note_reference = metadata.get("note_reference", [])
            
            # Håndter forskellige formater for note_reference
            if isinstance(note_reference, list):
                for ref in note_reference:
                    if isinstance(ref, dict):
                        ref_para = ref.get("paragraph", "")
                        ref_stykke = ref.get("stykke", "")
                        ref_key = f"{ref_para}"
                        if ref_stykke:
                            ref_key += f" {ref_stykke}"
                        if ref_key in relevant_paragraphs:
                            related_notes.append({"chunk": chunk, "score": 5.0})  # Høj prioritet
                    elif isinstance(ref, str) and ref in relevant_paragraphs:
                        related_notes.append({"chunk": chunk, "score": 5.0})
            elif isinstance(note_reference, str) and note_reference in relevant_paragraphs:
                related_notes.append({"chunk": chunk, "score": 5.0})
    
    # 6. Check også for noter refereret i fortolkningsbidrag
    for result in semantic_results + metadata_results:
        metadata = result["chunk"].get("metadata", {})
        if not metadata.get("is_note", False) and "fortolkningsbidrag" in metadata:
            fortolkningsbidrag = metadata.get("fortolkningsbidrag", [])
            for note_num in fortolkningsbidrag:
                # Find denne note i chunks
                for chunk in chunks:
                    if (chunk.get("metadata", {}).get("is_note", False) and 
                        str(chunk.get("metadata", {}).get("note_number", "")) == str(note_num)):
                        if not any(r["chunk"] == chunk for r in related_notes):
                            related_notes.append({"chunk": chunk, "score": 5.0})
    
    # 7. Kombiner alle resultater
    all_results = []
    
    # Tilføj metadata-resultater først
    for result in metadata_results:
        all_results.append(result)
    
    # Tilføj semantiske resultater, men undgå dubletter
    for result in semantic_results:
        if not any(r["chunk"]["content"] == result["chunk"]["content"] for r in all_results):
            all_results.append(result)
    
    # Tilføj relaterede noter, men undgå dubletter
    for note in related_notes:
        if not any(r["chunk"]["content"] == note["chunk"]["content"] for r in all_results):
            all_results.append(note)
    
    # Sortér efter score
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    return all_results[:top_k]

def identify_legal_concepts(query):
    """
    Identificerer juridiske koncepter i spørgsmålet.
    
    Args:
        query: Søgetekst/spørgsmål
    
    Returns:
        Dictionary med identificerede koncepter
    """
    concepts = {
        "paragraphs": [],
        "notes": [],
        "themes": [],
        "groups": [],
        "special_rules": []
    }
    
    # Identificer paragraffer og stykker
    paragraph_pattern = re.compile(r'(?:§|LL)\s*(\d+\s*[A-Za-z]?)(?:,?\s*stk\.?\s*(\d+))?', re.IGNORECASE)
    paragraph_matches = paragraph_pattern.findall(query)
    
    for match in paragraph_matches:
        paragraph_num = match[0].strip()
        stykke_num = match[1].strip() if len(match) > 1 and match[1] else None
        
        paragraph = f"§ {paragraph_num}"
        if stykke_num:
            concepts["paragraphs"].append((paragraph, f"Stk. {stykke_num}"))
        else:
            concepts["paragraphs"].append((paragraph, None))
    
    # Identificer noter
    note_pattern = re.compile(r'note\s*(\d+)', re.IGNORECASE)
    note_matches = note_pattern.findall(query)
    
    for match in note_matches:
        concepts["notes"].append(match)
    
    # Identificer temaer
    themes = ["dobbeltbeskatning", "lempelse", "skattefritagelse", "skattepligt", "udlandsophold", 
              "grænsegænger", "systemeksport", "offentligt ansat", "søfolk"]
    
    for theme in themes:
        if re.search(r'\b' + theme + r'\b', query.lower()):
            concepts["themes"].append(theme)
    
    # Identificer persongrupper
    groups = ["grænsegænger", "offentligt ansat", "søfolk", "udsendt", "selvstændig"]
    
    for group in groups:
        if re.search(r'\b' + group + r'\b', query.lower()):
            concepts["groups"].append(group)
    
    # Identificer specialregler/undtagelser
    special_rules = ["undtagelse", "særregel", "halv lempelse", "fuldt skattepligtig"]
    
    for rule in special_rules:
        if re.search(r'\b' + rule + r'\b', query.lower()):
            concepts["special_rules"].append(rule)
    
    return concepts

def filter_chunks_by_metadata(query, chunks, concepts):
    """
    Filtrerer chunks baseret på metadata i forhold til identificerede koncepter.
    
    Args:
        query: Søgetekst/spørgsmål
        chunks: Liste af chunks
        concepts: Dictionary med identificerede koncepter
    
    Returns:
        Liste af filtrerede chunks med scores
    """
    filtered_chunks = []
    
    # Tjek for paragraffer nævnt i spørgsmålet
    for paragraph, stykke in concepts["paragraphs"]:
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if metadata.get("paragraph") == paragraph:
                if stykke is None or metadata.get("stykke") == stykke:
                    filtered_chunks.append({"chunk": chunk, "score": 10.0})  # Høj score for direkte match
    
    # Tjek for noter nævnt i spørgsmålet
    for note_num in concepts["notes"]:
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if metadata.get("is_note", False) and str(metadata.get("note_number", "")) == note_num:
                filtered_chunks.append({"chunk": chunk, "score": 10.0})
    
    # Tjek for temaer nævnt i spørgsmålet
    for theme in concepts["themes"]:
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if (theme.lower() in (metadata.get("theme", "") or "").lower() or 
                theme.lower() in (metadata.get("subtheme", "") or "").lower()):
                filtered_chunks.append({"chunk": chunk, "score": 7.0})
    
    # Tjek for persongrupper nævnt i spørgsmålet
    for group in concepts["groups"]:
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            affected_groups = metadata.get("affected_groups", [])
            if any(group.lower() in ag.lower() for ag in affected_groups):
                filtered_chunks.append({"chunk": chunk, "score": 7.0})
    
    # Fjern dubletter (behold højeste score)
    unique_chunks = {}
    for item in filtered_chunks:
        chunk_id = item["chunk"].get("content", "")[:50]  # Brug starten af indholdet som ID
        if chunk_id not in unique_chunks or unique_chunks[chunk_id]["score"] < item["score"]:
            unique_chunks[chunk_id] = item
    
    return list(unique_chunks.values())