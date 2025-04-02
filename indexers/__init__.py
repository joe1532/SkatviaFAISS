import importlib
import os
import inspect

def get_available_indexers():
    """
    Returnerer ordbog over alle tilgængelige indekserere.
    
    Returns:
        dict: Ordbog med indekserer-ID som nøgle og metadata som værdi
    """
    indexers = {
        "lovtekst": {
            "display_name": "Lovbekendtgørelse",
            "description": "Indeksering af lovtekster med noter og krydsreferencer"
        },
        "vejledning": {
            "display_name": "Vejledning",
            "description": "Indeksering af skatteretlige vejledninger"
        },
        "cirkulaere": {
            "display_name": "Cirkulære",
            "description": "Indeksering af cirkulærer"
        },
        "afgoerelse": {
            "display_name": "Afgørelse",
            "description": "Indeksering af domme og afgørelser"
        },
        "juridisk_vejledning": {
            "display_name": "Den Juridiske Vejledning",
            "description": "Specialiseret indeksering af Den Juridiske Vejledning med underafsnit og referencer"
        },
        "generisk": {
            "display_name": "Generisk dokument",
            "description": "Generisk indeksering af dokumenter der ikke passer i andre kategorier"
        }
    }
    
    return indexers

def get_indexer_class(indexer_type):
    """
    Returnerer indeksererklassen for den angivne type.
    
    Args:
        indexer_type (str): Type af indekserer
        
    Returns:
        class: Indeksererklasse
        
    Raises:
        ImportError: Hvis indeksereren ikke kan importeres
    """
    if not indexer_type:
        indexer_type = "generisk"
    
    try:
        # Forsøg at importere det specifikke indekseringsmodul
        module_name = f"{indexer_type}_indexer"
        module = importlib.import_module(f".{module_name}", package="indexers")
        
        # Find Indexer-klassen i modulet
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name == "Indexer":
                return obj
                
        raise ImportError(f"Ingen Indexer-klasse fundet i {module_name}")
        
    except ImportError as e:
        # Hvis den specifikke indekserer ikke findes, brug den generiske
        if indexer_type != "generisk":
            print(f"Kunne ikke indlæse {indexer_type}_indexer: {e}. Bruger generisk_indexer i stedet.")
            return get_indexer_class("generisk")
        else:
            raise ImportError(f"Kunne ikke indlæse generisk_indexer: {e}")