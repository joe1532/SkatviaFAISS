import os
import json
import hashlib
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
import re

def ensure_cache_directory(cache_dir="cache"):
    """Sikrer at cache-mappen eksisterer."""
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def cached_call_gpt4o(prompt, model="gpt-4o", json_mode=True, cache_dir="cache"):
    """
    Kalder GPT-4o med caching for at undgå gentagne API-kald.
    
    Args:
        prompt: Teksten der sendes til modellen
        model: Modelnavn ("gpt-4o" eller "gpt-3.5-turbo")
        json_mode: Om svaret skal være i JSON-format
        cache_dir: Mappe til at gemme cache-filer
        
    Returns:
        JSON-objekt eller tekst fra modellen (cachelagret hvis tilgængelig)
    """
    from utils import api_utils  # Importér her for at undgå cirkulære importer
    
    # Sikr at cache-mappen eksisterer
    ensure_cache_directory(cache_dir)
    
    # Generér en unik nøgle baseret på prompt og model
    # Brug kun de første 10000 tegn af prompten for store dokumenter
    hash_input = f"{prompt[:10000]}{model}{json_mode}".encode('utf-8')
    cache_key = hashlib.md5(hash_input).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Tjek om resultatet allerede er cachet
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                st.info("Bruger cachelagret resultat")
                
                # Tæl cache hits hvis attributten eksisterer
                if hasattr(cached_call_gpt4o, 'cache_hits'):
                    cached_call_gpt4o.cache_hits += 1
                else:
                    cached_call_gpt4o.cache_hits = 1
                    
                return json.load(f)
        except Exception as e:
            st.warning(f"Kunne ikke indlæse cache: {e}")
    
    # Hvis ikke cachet, kald API'et
    # Tæl cache misses hvis attributten eksisterer
    if hasattr(cached_call_gpt4o, 'cache_misses'):
        cached_call_gpt4o.cache_misses += 1
    else:
        cached_call_gpt4o.cache_misses = 1
        
    result = api_utils.call_gpt4o(prompt, model=model, json_mode=json_mode)
    
    # Gem resultatet i cache
    if result:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.warning(f"Kunne ikke gemme cache: {e}")
    
    return result

def process_segments_parallel(segments, doc_type_key, context_summary, doc_id, options, get_template_func=None):
    """
    Behandler segmenter parallelt med begrænset samtidighed og forbedret fejlhåndtering.
    
    Args:
        segments: Liste af tekstsegmenter
        doc_type_key: Nøgle til dokumenttype
        context_summary: Kontekstopsummering
        doc_id: Dokument-id
        options: Processeringsindstillinger
        get_template_func: Funktion til at hente indexing prompt template
        
    Returns:
        Liste af chunks fra alle segmenter
    """
    from utils import api_utils  # Importér her for at undgå cirkulære importer
    
    # Brug den medfølgende funktion i stedet for at importere
    if get_template_func is None:
        st.error("Mangler get_template_func parameter")
        return []
    
    # Begræns segmentlængde og del op hvis nødvendigt
    max_segment_len = 15000  # Maksimal sikker segmentlængde
    processed_segments = []
    for i, segment in enumerate(segments):
        if len(segment) > max_segment_len:
            st.warning(f"Segment {i+1} er for langt ({len(segment)} tegn). Opdeler det i mindre dele.")
            # Del segmentet op med semantisk forståelse
            parts = split_segment_semantically(segment, max_segment_len)
            processed_segments.extend(parts)
            st.info(f"Segment {i+1} opdelt i {len(parts)} dele.")
        else:
            processed_segments.append(segment)
    
    segments = processed_segments
    st.info(f"Total {len(segments)} segmenter at behandle efter opdeling.")
    
    def process_single_segment(segment_info):
        segment, segment_idx = segment_info
        time.sleep(segment_idx * 1.0)  # Længere ventetid mellem kald
        
        try:
            # Hent indekseringsprompt med den medfølgte funktion
            if hasattr(get_template_func, '__self__'):  # Det er en objektmetode
                indexing_prompt = get_template_func(doc_type_key, context_summary, doc_id, segment_idx+1)
            else:  # Det er en normal funktion
                indexing_prompt = get_template_func(doc_type_key, context_summary, doc_id, segment_idx+1)
            
            if not indexing_prompt or len(indexing_prompt) < 10:
                st.error(f"Ugyldig prompt for segment {segment_idx+1}. Prompt er for kort eller tom.")
                return {"chunks": []}
            
            # Tilføj teksten til prompten
            indexing_prompt_with_text = indexing_prompt + f"\n\nDokument (del {segment_idx+1}):\n" + segment
            
            # Sikr at vi bruger JSON-mode
            if "RETURNER DIN SVAR SOM JSON" not in indexing_prompt_with_text:
                indexing_prompt_with_text += "\n\nRETURNER DIN SVAR SOM JSON."
            
            # Direkte kald til API i stedet for cached_call_gpt4o for mere kontrol
            result = api_utils.call_gpt4o(
                indexing_prompt_with_text, 
                model=options.get("model", "gpt-4o"),
                json_mode=True
            )
            
            if not result:
                st.error(f"Intet resultat for segment {segment_idx+1}.")
                return {"chunks": []}
            
            # Tjek resultatet
            if isinstance(result, dict):
                if "chunks" in result:
                    # Tilføj segment position til hvert chunk for sortering og kontekst
                    for chunk in result["chunks"]:
                        if "metadata" in chunk:
                            chunk["metadata"]["segment_position"] = segment_idx
                            chunk["metadata"]["segment_count"] = len(segments)
                    return result
                else:
                    st.warning(f"Segment {segment_idx+1}: Resultat indeholder ikke 'chunks'. Nøgler: {list(result.keys())}")
                    # Forsøg at tilpasse resultatformatet til forventet format
                    if "content" in result:
                        st.info(f"Segment {segment_idx+1}: Forsøger at udtrække chunks fra 'content'.")
                        try:
                            # Konverter til JSON igen hvis det er en streng
                            if isinstance(result["content"], str):
                                content_json = json.loads(result["content"])
                                if "chunks" in content_json:
                                    return content_json
                            return {"chunks": [{"content": result["content"], "metadata": {"segment_position": segment_idx}}]}
                        except Exception as e:
                            st.error(f"Kunne ikke udtrække chunks: {e}")
                    return {"chunks": []}
            elif isinstance(result, str):
                st.warning(f"Segment {segment_idx+1}: Resultat er en streng, ikke et JSON-objekt. Forsøger at parse.")
                try:
                    # Forsøg at udtrække JSON fra strengen
                    if "{" in result and "}" in result:
                        json_str = result[result.find("{"):result.rfind("}")+1]
                        json_obj = json.loads(json_str)
                        if "chunks" in json_obj:
                            # Tilføj segment position
                            for chunk in json_obj["chunks"]:
                                if "metadata" in chunk:
                                    chunk["metadata"]["segment_position"] = segment_idx
                                    chunk["metadata"]["segment_count"] = len(segments)
                            return json_obj
                    return {"chunks": [{"content": result, "metadata": {"segment_position": segment_idx}}]}
                except Exception as e:
                    st.error(f"Kunne ikke parse JSON fra streng: {e}")
                    return {"chunks": []}
            
            # Fallback
            return {"chunks": []}
            
        except Exception as e:
            st.error(f"Fejl ved behandling af segment {segment_idx+1}: {str(e)}")
            # Vis mere detaljerede fejloplysninger
            import traceback
            st.code(traceback.format_exc())
            return {"chunks": []}
    
    all_chunks = []
    segment_tuples = [(segment, i) for i, segment in enumerate(segments)]
    
    # Bearbejd hvert segment sekventielt for at undgå problemer med rate limits
    for i, segment_info in enumerate(segment_tuples):
        progress_pct = (i / len(segment_tuples)) * 100
        st.write(f"Behandler segment {i+1}/{len(segment_tuples)} ({progress_pct:.1f}%)...")
        
        segment_result = process_single_segment(segment_info)
        
        if segment_result and "chunks" in segment_result and segment_result["chunks"]:
            chunk_count = len(segment_result["chunks"])
            all_chunks.extend(segment_result["chunks"])
            st.success(f"Segment {i+1} behandlet: {chunk_count} chunks genereret")
        else:
            st.warning(f"Kunne ikke indeksere segment {i+1}. Fortsætter med næste segment.")
        
        # Vent mellem segmenter for at undgå rate limits
        if i < len(segment_tuples) - 1:
            wait_time = options.get("wait_time", 5)
            st.info(f"Venter {wait_time} sekunder før næste segment...")
            time.sleep(wait_time)
    
    # Vis det samlede resultat
    st.success(f"Behandling fuldført. Genereret {len(all_chunks)} chunks fra {len(segments)} segmenter.")
    
    # Sorter chunks efter position hvis metadata indeholder dette
    try:
        all_chunks.sort(key=lambda c: (
            c.get("metadata", {}).get("segment_position", 0),
            c.get("metadata", {}).get("chunk_position", 0)
        ))
    except Exception as e:
        st.warning(f"Kunne ikke sortere chunks: {e}")
    
    return all_chunks

def split_segment_semantically(segment, max_length=15000):
    """
    Deler et segment op på semantisk fornuftige steder med juridisk kontekst.
    
    Args:
        segment: Tekst at dele op
        max_length: Maksimal længde for et segment
        
    Returns:
        Liste af opdelte segmenter
    """
    # Hvis segmentet er kort nok, returner det uændret
    if len(segment) <= max_length:
        return [segment]
    
    parts = []
    
    # Prøv først at dele ved afsnitsoverskrifter
    markers = [
        r'\n\s*\n[A-ZÆØÅ][a-zæøåA-ZÆØÅ\s]+\n',  # Overskrift
        r'\n\s*\n\d+\.\s+[A-ZÆØÅ]',             # Nummereret afsnit
        r'\n\s*\nBemærk\s+',                    # Bemærk-sektion
        r'\n\s*\nEksempel\s+\d+:',              # Eksempel
        r'\n\s*\nSe også\s+'                    # Se også-sektion
    ]
    
    breakpoints = []
    for marker in markers:
        for match in re.finditer(marker, segment):
            breakpoints.append(match.start())
    
    # Sortér breakpoints
    breakpoints = sorted(set(breakpoints))
    
    # Hvis ingen semantiske breakpoints blev fundet, eller første er for langt inde
    if not breakpoints or breakpoints[0] > max_length:
        # Del ved afsnit
        paragraphs = segment.split('\n\n')
        
        current_part = ""
        for para in paragraphs:
            if not para.strip():  # Skip tomme afsnit
                continue
                
            if len(current_part + para + "\n\n") <= max_length:
                current_part += para + "\n\n"
            else:
                # Gem nuværende del og start ny
                if current_part:
                    parts.append(current_part.strip())
                    current_part = para + "\n\n"
                else:
                    # Paragraffen selv er for lang, del ved sætninger
                    sentences = []
                    for sentence in re.split(r'(?<=[.!?])\s+', para):
                        if sentence.strip():
                            sentences.append(sentence)
                    
                    current_sentence_part = ""
                    for sentence in sentences:
                        if len(current_sentence_part + sentence + " ") <= max_length:
                            current_sentence_part += sentence + " "
                        else:
                            if current_sentence_part:
                                parts.append(current_sentence_part.strip())
                                current_sentence_part = sentence + " "
                            else:
                                # Sætningen selv er for lang, del vilkårligt
                                for j in range(0, len(sentence), max_length // 2):
                                    parts.append(sentence[j:j + max_length // 2].strip())
                    
                    if current_sentence_part:
                        current_part = current_sentence_part
        
        # Tilføj sidste del
        if current_part:
            parts.append(current_part.strip())
    else:
        # Del ved semantiske breakpoints
        start_pos = 0
        for bp in breakpoints:
            # Hvis denne del er større end max_length, del den yderligere
            if bp - start_pos > max_length:
                # Rekursiv opdeling af dette segment
                subsegment = segment[start_pos:bp]
                subparts = split_segment_semantically(subsegment, max_length)
                parts.extend(subparts)
            else:
                # Tilføj dette segment direkte
                part = segment[start_pos:bp].strip()
                if part:
                    parts.append(part)
            
            start_pos = bp
        
        # Tilføj sidste del
        if start_pos < len(segment):
            last_part = segment[start_pos:].strip()
            if len(last_part) <= max_length:
                if last_part:
                    parts.append(last_part)
            else:
                # Sidste del er for stor, del den rekursivt
                subparts = split_segment_semantically(last_part, max_length)
                parts.extend(subparts)
    
    return parts

def optimize_chunks(chunks):
    """
    Optimerer chunks for bedre søgning og reduceret redundans.
    
    Args:
        chunks: Liste af chunks
        
    Returns:
        Optimeret liste af chunks
    """
    if not chunks:
        return []
    
    # 1. Fjern tomme chunks
    non_empty_chunks = [c for c in chunks if c.get("content", "").strip()]
    
    # 2. Fjern duplikater baseret på indhold
    unique_chunks = []
    content_hashes = set()
    
    for chunk in non_empty_chunks:
        # Hash af de første 100 tegn + længde (for at undgå kollisioner på små tekster)
        content = chunk.get("content", "")
        content_hash = hash(content[:100] + str(len(content)))
        
        if content_hash not in content_hashes:
            content_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    # 3. Sørg for at metadata er komplet
    standardized_chunks = []
    
    for chunk in unique_chunks:
        # Sikr at metadata eksisterer
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        
        # Basisfelter der bør eksistere i alle chunks
        required_fields = {
            "concepts": [],
            "law_references": [],
            "case_references": [],
            "affected_groups": [],
            "legal_exceptions": [],
            "theme": "",
            "subtheme": "",
            "is_example": False,
            "complexity": "moderat",
            "chunk_type": "text"
        }
        
        # Tilføj manglende felter
        for field, default_value in required_fields.items():
            if field not in chunk["metadata"]:
                chunk["metadata"][field] = default_value
        
        standardized_chunks.append(chunk)
    
    # 4. Tilføj retrievability score
    for chunk in standardized_chunks:
        # Beregn en score baseret på metadata-rigdom og chunkstørrelse
        score = 0.5  # Base score
        
        # +0.1 for hver relevant metadata-type der findes
        if chunk["metadata"].get("law_references"):
            score += 0.1
        if chunk["metadata"].get("case_references"):
            score += 0.1
        if len(chunk["metadata"].get("concepts", [])) >= 3:
            score += 0.1
        
        # Størrelse: ideel størrelse er 800-1500 tegn
        length = len(chunk.get("content", ""))
        if 800 <= length <= 1500:
            score += 0.2
        elif length < 200:
            score -= 0.2  # For små chunks er mindre brugbare
        elif length > 3000:
            score -= 0.1  # For store chunks er sværere at søge i
        
        # Eksempler er ofte nyttige søgeresultater
        if chunk["metadata"].get("is_example"):
            score += 0.1
        
        # Normalisér scoren til 0-1 området
        score = max(0.0, min(1.0, score))
        chunk["metadata"]["retrievability_score"] = score
    
    # 5. Organisér chunks i logisk rækkefølge hvis muligt
    if all("segment_position" in c["metadata"] for c in standardized_chunks):
        # Sorter efter segment position og derefter efter eventuelt chunk position
        standardized_chunks.sort(key=lambda c: (
            c["metadata"]["segment_position"],
            c["metadata"].get("chunk_position", 0)
        ))
    
    return standardized_chunks

def merge_small_chunks(chunks, min_size=200, target_size=1000):
    """
    Slår for små chunks sammen til større chunks baseret på semantisk sammenhæng.
    
    Args:
        chunks: Liste af chunks at behandle
        min_size: Minimum ønsket størrelse for et chunk
        target_size: Målstørrelse for chunks
        
    Returns:
        Liste af chunks med sammenslåede små chunks
    """
    # Identificér små chunks
    small_chunks = [c for c in chunks if len(c.get("content", "")) < min_size]
    normal_chunks = [c for c in chunks if len(c.get("content", "")) >= min_size]
    
    if not small_chunks:
        return chunks  # Ingen små chunks at behandle
    
    # Gruppér små chunks baseret på afsnit og underafsnit
    section_groups = {}
    for chunk in small_chunks:
        metadata = chunk.get("metadata", {})
        section = metadata.get("section", "unknown")
        subsection = metadata.get("subsection", "")
        
        key = (section, subsection)
        if key not in section_groups:
            section_groups[key] = []
        section_groups[key].append(chunk)
    
    # For hver gruppe, slå chunks sammen hvis de tilsammen er under målstørrelsen
    merged_chunks = []
    
    for key, group in section_groups.items():
        # Sortér gruppen efter position hvis tilgængelig
        group.sort(key=lambda c: (
            c.get("metadata", {}).get("segment_position", 0),
            c.get("metadata", {}).get("chunk_position", 0)
        ))
        
        current_content = ""
        current_metadata = None
        
        for chunk in group:
            if not current_metadata:
                current_metadata = chunk.get("metadata", {}).copy()
            
            # Hvis tilføjelse af denne chunk holder os under målstørrelsen, tilføj den
            if len(current_content + "\n\n" + chunk.get("content", "")) <= target_size:
                if current_content:
                    current_content += "\n\n"
                current_content += chunk.get("content", "")
                
                # Kombinér metadata lister
                for field in ["concepts", "law_references", "case_references", "affected_groups", "legal_exceptions"]:
                    if field in chunk.get("metadata", {}) and field in current_metadata:
                        combined = list(set(current_metadata[field] + chunk.get("metadata", {}).get(field, [])))
                        current_metadata[field] = combined
            else:
                # Denne chunk ville overstige målstørrelsen, gem den aktuelle og start en ny
                if current_content:
                    merged_chunks.append({
                        "content": current_content,
                        "metadata": current_metadata
                    })
                    
                    current_content = chunk.get("content", "")
                    current_metadata = chunk.get("metadata", {}).copy()
                else:
                    # Behold denne chunk som den er
                    merged_chunks.append(chunk)
        
        # Tilføj sidste sammenslåede chunk
        if current_content and current_metadata:
            merged_chunks.append({
                "content": current_content,
                "metadata": current_metadata
            })
    
    # Kombinér de sammenslåede små chunks med de normale chunks
    result = normal_chunks + merged_chunks
    
    # Opdater retrievability score
    for chunk in result:
        if "metadata" in chunk:
            # Beregn en simpel score baseret på metadata-rigdom og chunklængde
            score = 0.5  # Base score
            
            # +0.1 for hver relevant metadata-type der findes
            if chunk["metadata"].get("law_references"):
                score += 0.1
            if chunk["metadata"].get("case_references"):
                score += 0.1
            if len(chunk["metadata"].get("concepts", [])) >= 3:
                score += 0.1
            
            # Størrelse: ideel størrelse er 800-1500 tegn
            length = len(chunk.get("content", ""))
            if 800 <= length <= 1500:
                score += 0.2
            elif length < 200:
                score -= 0.2  # For små chunks er mindre brugbare
            elif length > 3000:
                score -= 0.1  # For store chunks er sværere at søge i
            
            # Eksempler er ofte nyttige søgeresultater
            if chunk["metadata"].get("is_example"):
                score += 0.1
            
            # Normalisér scoren til 0-1 området
            score = max(0.0, min(1.0, score))
            chunk["metadata"]["retrievability_score"] = score
    
    return result

def split_large_chunks(chunks, max_size=2000):
    """
    Opdeler for store chunks i mindre dele med respekt for semantisk sammenhæng.
    
    Args:
        chunks: Liste af chunks at behandle
        max_size: Maksimal ønsket størrelse for et chunk
        
    Returns:
        Liste af chunks med opdelte store chunks
    """
    # Identificér store chunks
    large_chunks = [c for c in chunks if len(c.get("content", "")) > max_size]
    normal_chunks = [c for c in chunks if len(c.get("content", "")) <= max_size]
    
    if not large_chunks:
        return chunks  # Ingen store chunks at opdele
    
    # Opdel de store chunks
    split_chunks = []
    
    for chunk in large_chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {}).copy()
        
        # Del ved afsnit
        paragraphs = content.split("\n\n")
        
        if len(paragraphs) <= 1 or max(len(p) for p in paragraphs) > max_size:
            # Kan ikke dele ved afsnit eller afsnit er selv for store, del ved sætningsgrænser
            sentences = []
            for para in paragraphs:
                sentences.extend(re.split(r'(?<=[.!?])\s+', para))
            
            current_content = ""
            for sentence in sentences:
                if len(current_content + sentence + " ") <= max_size:
                    current_content += sentence + " "
                else:
                    if current_content:
                        # Lav et nyt chunk
                        new_metadata = metadata.copy()
                        new_metadata["chunk_id"] = f"{metadata.get('chunk_id', 'chunk')}_{len(split_chunks)}"
                        split_chunks.append({
                            "content": current_content.strip(),
                            "metadata": new_metadata
                        })
                    
                    current_content = sentence + " "
            
            # Tilføj sidste del
            if current_content:
                new_metadata = metadata.copy()
                new_metadata["chunk_id"] = f"{metadata.get('chunk_id', 'chunk')}_{len(split_chunks)}"
                split_chunks.append({
                    "content": current_content.strip(),
                    "metadata": new_metadata
                })
        
        else:
            # Del ved afsnitsgrænser
            current_content = ""
            for para in paragraphs:
                if not para.strip():  # Skip tomme afsnit
                    continue
                    
                if len(current_content + para + "\n\n") <= max_size:
                    current_content += para + "\n\n"
                else:
                    if current_content:
                        # Lav et nyt chunk
                        new_metadata = metadata.copy()
                        new_metadata["chunk_id"] = f"{metadata.get('chunk_id', 'chunk')}_{len(split_chunks)}"
                        split_chunks.append({
                            "content": current_content.strip(),
                            "metadata": new_metadata
                        })
                    
                    current_content = para + "\n\n"
            
            # Tilføj sidste del
            if current_content:
                new_metadata = metadata.copy()
                new_metadata["chunk_id"] = f"{metadata.get('chunk_id', 'chunk')}_{len(split_chunks)}"
                split_chunks.append({
                    "content": current_content.strip(),
                    "metadata": new_metadata
                })
    
    # Kombinér de opdelte chunks med de normale chunks
    result = normal_chunks + split_chunks
    
    # Opdater retrievability score
    for chunk in result:
        if "metadata" in chunk:
            # Beregn en simpel score baseret på metadata-rigdom og chunklængde
            score = 0.5  # Base score
            
            # +0.1 for hver relevant metadata-type der findes
            if chunk["metadata"].get("law_references"):
                score += 0.1
            if chunk["metadata"].get("case_references"):
                score += 0.1
            if len(chunk["metadata"].get("concepts", [])) >= 3:
                score += 0.1
            
            # Størrelse: ideel størrelse er 800-1500 tegn
            length = len(chunk.get("content", ""))
            if 800 <= length <= 1500:
                score += 0.2
            elif length < 200:
                score -= 0.2  # For små chunks er mindre brugbare
            elif length > 3000:
                score -= 0.1  # For store chunks er sværere at søge i
            
            # Eksempler er ofte nyttige søgeresultater
            if chunk["metadata"].get("is_example"):
                score += 0.1
            
            # Normalisér scoren
            score = max(0.0, min(1.0, score))
            chunk["metadata"]["retrievability_score"] = score
    
    return result