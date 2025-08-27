from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai.tools import BaseTool, tool # Import corretto da crewai.tools
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS = None
    DDGS_AVAILABLE = False

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

# Tool personalizzato per DuckDuckGo
class CustomDuckDuckGoSearchToolSchema(BaseModel):
    """Schema per l'input del tool di ricerca"""
    query: str = Field(description="La query di ricerca da eseguire su DuckDuckGo")

class CustomDuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Cerca informazioni su internet usando DuckDuckGo e restituisce i primi risultati"
    args_schema: Type[BaseModel] = CustomDuckDuckGoSearchToolSchema
    
    def _run(self, query: str) -> str:
        """Esegue una ricerca su DuckDuckGo e restituisce i risultati formattati"""
        
        # Controllo se la libreria è disponibile
        if not DDGS_AVAILABLE:
            return ("Errore: libreria duckduckgo-search non installata.\n"
                    "Installa con: pip install duckduckgo-search")
        
        try:
            results = []
            with DDGS(verify=False) as ddgs:
                # Cerca fino a 3 risultati
                search_results = list(ddgs.text(
                    keywords=query,  # Usa keywords invece di query per DDGS
                    region='it-it', 
                    safesearch='off', 
                    max_results=3
                ))
                
                for i, r in enumerate(search_results, 1):
                    title = r.get("title", "")
                    url = r.get("href", "") or r.get("link", "")
                    snippet = r.get("body", "")
                    
                    result_text = f"""
Risultato {i}:
Titolo: {title}
URL: {url}
Snippet: {snippet}
"""
                    results.append(result_text)
            
            if results:
                return "\n".join(results)
            else:
                return f"Nessun risultato trovato per la query: {query}"
                
        except Exception as e:
            return f"Errore durante la ricerca: {str(e)}"


# Definisci il tool di somma con docstring chiara (CrewAI userà questa descrizione)
@tool
def add_numbers(a: int, b: int) -> int:
    """
    add_numbers(a: int, b: int) -> int
    Restituisce la somma di due numeri interi passati come parametri.
    Parametri:
      - a: primo addendo (int)
      - b: secondo addendo (int)
    Ritorno:
      - int: somma di a e b
    """
    return a + b
