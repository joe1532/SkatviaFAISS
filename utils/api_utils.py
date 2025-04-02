import os
import json
import time
from openai import OpenAI
import streamlit as st

@st.cache_resource
def get_openai_client():
    """Henter OpenAI-klienten baseret på miljøvariabel eller Streamlit secrets."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY ikke fundet i miljøvariablerne eller Streamlit secrets")
    return OpenAI(api_key=api_key)

def call_gpt4o(prompt, model="gpt-4o", json_mode=True, max_retries=3, retry_delay=10):
    """
    Kalder GPT-4o med håndtering af rate limits og fejl.
    
    Args:
        prompt: Teksten der sendes til modellen
        model: Modelnavn ("gpt-4o" eller "gpt-3.5-turbo")
        json_mode: Om svaret skal være i JSON-format
        max_retries: Maksimalt antal forsøg ved fejl
        retry_delay: Ventetid mellem forsøg (i sekunder)
        
    Returns:
        JSON-objekt eller tekst fra modellen
    """
    client = get_openai_client()
    
    # Tilføj json-reference i prompten hvis json_mode er aktiveret
    if json_mode:
        # Tjek om json allerede er nævnt i prompten
        if "json" not in prompt.lower() and "JSON" not in prompt:
            if "RETURNER DIN SVAR SOM JSON" not in prompt:
                prompt = prompt + "\n\nRETURNER DIN SVAR SOM JSON."
    
    for attempt in range(max_retries):
        try:
            # Log første 100 tegn af prompten for debugging
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            st.info(f"Kalder API med prompt (forkortet): {prompt_preview}")
            
            # For debugging, log om json_mode er aktiveret
            if json_mode:
                st.info(f"JSON-mode er aktiveret. Model: {model}")
            
            messages = [{"role": "user", "content": prompt}]
            response_format = {"type": "json_object"} if json_mode else None
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # Log om vi fik et svar
            st.info(f"Svar modtaget fra API. Længde: {len(content)} tegn")
            
            if json_mode:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    st.warning(f"JSON decode fejl: {str(e)}. Forsøger at reparere JSON...")
                    # Simpel reparation af JSON-fejl
                    content = content.strip()
                    if not content.startswith('{'):
                        content = '{' + content.split('{', 1)[1]
                    if not content.endswith('}'):
                        content = content.rsplit('}', 1)[0] + '}'
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError as e2:
                        st.error(f"Kunne ikke reparere JSON: {str(e2)}")
                        # Vis starten af indholdet for fejlsøgning
                        st.code(content[:500] + "..." if len(content) > 500 else content)
                        return {"error": "JSON parse error", "content": content}
            
            return content
            
        except Exception as e:
            error_message = str(e)
            
            # Særlig håndtering af response_format fejl
            if "response_format" in error_message and "json" in error_message:
                st.warning("Fejl med JSON format. Forsøger igen uden JSON mode...")
                # Deaktiver json_mode og forsøg igen
                return call_gpt4o(prompt, model=model, json_mode=False, max_retries=max_retries-1, retry_delay=retry_delay)
            
            # Håndtering af rate limit errors
            if "rate_limit_exceeded" in error_message and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Eksponentiel backoff
                st.warning(f"Rate limit overskredet. Venter {wait_time} sekunder før næste forsøg...")
                time.sleep(wait_time)
            else:
                st.error(f"Fejl ved kald til OpenAI: {e}")
                return None

def generate_embedding(text, max_retries=3, retry_delay=5):
    """
    Genererer embedding for en tekst med håndtering af rate limits.
    
    Args:
        text: Teksten der skal embeddes
        max_retries: Maksimalt antal forsøg ved fejl
        retry_delay: Ventetid mellem forsøg (i sekunder)
        
    Returns:
        Embedding-vektor eller None ved fejl
    """
    client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                time.sleep(wait_time)
            else:
                st.error(f"Fejl ved generering af embedding: {e}")
                return None

def estimate_tokens(text):
    """
    Estimerer antallet af tokens i en tekst (groft estimat).
    
    Args:
        text: Teksten der skal estimeres
        
    Returns:
        Estimeret antal tokens
    """
    # Ca. 4 tegn per token for dansk tekst (groft estimat)
    return len(text) // 4