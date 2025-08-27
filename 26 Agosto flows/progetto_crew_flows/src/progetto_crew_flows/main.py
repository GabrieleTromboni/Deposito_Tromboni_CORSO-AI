import sys
import json
import requests

# Importa i componenti reali di CrewAI
from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, listen, router
from tools.custom_tool import CustomDuckDuckGoSearchTool  # per ricerca web avanzata

try:
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("CrewAI non disponibile. Installalo con: pip install crewai crewai-tools")

# Provo a importare il decoratore tool di crewai; se non presente, uso un fallback no-op
try:
    from crewai.tools import tool  # type: ignore
except Exception:
    def tool(func=None, **kwargs):
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func

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

def web_search(query: str) -> str:
    return CustomDuckDuckGoSearchTool().search(query)

def prompt_choice(prompt: str, choices):
    choice = input(prompt).strip().lower()
    if choice in choices:
        return choice
    return None

def ask_int(prompt_text: str):
    while True:
        val = input(prompt_text).strip()
        try:
            return int(val)
        except ValueError:
            print("Per favore inserisci un numero intero valido.")

# --- Crew reali con Agent e Task ---

def create_search_crew():
    """Crea una Crew per la ricerca web con Agent e Task."""
    if not CREWAI_AVAILABLE:
        # Fallback se CrewAI non disponibile
        class MockCrew:
            def kickoff(self, inputs=None):
                query = inputs.get("query", "") if inputs else ""
                return {"result": web_search(query)}
        return MockCrew()
    
    # Agent ricercatore
    researcher = Agent(
        role="Web Researcher",
        goal="Find and summarize information from the web",
        backstory="You are an expert at finding relevant information online",
        verbose=True,
        allow_delegation=False,
        tools=[],  # Useremo web_search come funzione custom
    )
    
    # Task di ricerca
    search_task = Task(
        description="Search for: {query}",
        expected_output="A comprehensive summary of the search results",
        agent=researcher,
        tools=[],
        # Usiamo una funzione custom invece di tool esterni
        action=lambda inputs: web_search(inputs.get("query", ""))
    )
    
    # Crew di ricerca
    search_crew = Crew(
        agents=[researcher],
        tasks=[search_task],
        verbose=True,
        process="sequential"
    )
    
    return search_crew

def create_math_crew():
    """Crea una Crew per operazioni matematiche con Agent e Task."""
    if not CREWAI_AVAILABLE:
        # Fallback se CrewAI non disponibile
        class MockCrew:
            def kickoff(self, inputs=None):
                a = inputs.get("a", 0) if inputs else 0
                b = inputs.get("b", 0) if inputs else 0
                return {"result": add_numbers(a, b)}
        return MockCrew()
    
    # Agent matematico
    mathematician = Agent(
        role="Mathematician",
        goal="Perform mathematical calculations accurately",
        backstory="You are an expert mathematician who can solve any numerical problem",
        verbose=True,
        allow_delegation=False,
        tools=[add_numbers]  # Usa il tool add_numbers definito sopra
    )
    
    # Task di calcolo
    math_task = Task(
        description="Calculate the sum of {a} and {b}",
        expected_output="The exact sum of the two numbers",
        agent=mathematician,
        tools=[add_numbers]
    )
    
    # Crew matematica
    math_crew = Crew(
        agents=[mathematician],
        tasks=[math_task],
        verbose=True,
        process="sequential"
    )
    
    return math_crew

# --- Flow per orchestrare le Crew (opzionale) ---

class AgentFlow(Flow):
    """Flow che può eseguire ricerca o calcoli basandosi sull'input."""
    
    @start()
    def get_operation(self):
        self.state["operation"] = input("Vuoi fare una ricerca web o una somma? (digita 'ricerca' o 'somma'): ").strip().lower()

    @router(get_operation)
    def route_request(self):
        """Determina quale operazione eseguire."""
        operation = self.state.get("operation")
        if operation == "search":
            return "search"
        elif operation == "math":
            return "math"
        return "invalid"
    
    @listen("search")
    def perform_search(self):
        """Esegue la ricerca web."""
        search_crew = create_search_crew()
        query = self.state.get("query", "")
        result = search_crew.kickoff(inputs={"query": query})
        self.state["result"] = result
        return result
    
    @listen("math") 
    def perform_math(self):
        """Esegue il calcolo matematico."""
        math_crew = create_math_crew()
        a = self.state.get("a", 0)
        b = self.state.get("b", 0)
        result = math_crew.kickoff(inputs={"a": a, "b": b})
        self.state["result"] = result
        return result

def main():
    print("Agent interattivo con CrewAI: scegli 'ricerca' per una ricerca web o 'somma' per sommare due numeri.")
    
    # Crea le crew una volta sola
    search_crew = create_search_crew()
    math_crew = create_math_crew()
    
    while True:
        choice = None
        while choice is None:
            raw = input("Vuoi fare una ricerca web o una somma? (digita 'ricerca' o 'somma', 'esci' per terminare): ").strip().lower()
            if raw in ("ricerca", "somma", "esci"):
                choice = raw
            else:
                print("Scelta non valida. Digita 'ricerca', 'somma' o 'esci'.")
        
        if choice == "esci":
            print("Uscita. Ciao.")
            return
            
        if choice == "ricerca":
            query = input("Inserisci la query di ricerca: ").strip()
            print("\n=== Eseguo la ricerca con SearchCrew ===")
            try:
                if CREWAI_AVAILABLE:
                    result = search_crew.kickoff(inputs={"query": query})
                    if isinstance(result, dict):
                        print(f"\nRisultato: {result.get('result', result)}")
                    else:
                        print(f"\nRisultato: {result}")
                else:
                    # Fallback diretto
                    result = web_search(query)
                    print(f"\nRisultato per '{query}':\n{result}")
            except Exception as e:
                print(f"Errore durante la ricerca: {e}")
                
        else:  # somma
            a = ask_int("Inserisci il primo numero (a): ")
            b = ask_int("Inserisci il secondo numero (b): ")
            print("\n=== Calcolo la somma con MathCrew ===")
            try:
                if CREWAI_AVAILABLE:
                    result = math_crew.kickoff(inputs={"a": a, "b": b})
                    if isinstance(result, dict):
                        print(f"\nRisultato: la somma di {a} e {b} è {result.get('result', result)}")
                    else:
                        print(f"\nRisultato: la somma di {a} e {b} è {result}")
                else:
                    # Fallback diretto
                    s = add_numbers(a, b)
                    print(f"\nRisultato: la somma di {a} e {b} è {s}.")
            except Exception as e:
                print(f"Errore durante il calcolo: {e}")
        
        # chiedi se continuare
        cont = input("\nVuoi fare un'altra operazione? (s/n): ").strip().lower()
        if cont not in ("s", "si", "y", "yes"):
            print("Fine. Ciao.")
            break

if __name__ == "__main__":
    main()