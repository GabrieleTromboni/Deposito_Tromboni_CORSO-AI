"""
SearchCrew - Crew specializzata per la ricerca web.
Utilizza agent researcher con tool di ricerca DuckDuckGo personalizzato.
"""

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import sys
import os

# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.custom_tool import CustomDuckDuckGoSearchTool

# Define our models for structured data
class Section(BaseModel):
    title: str = Field(description="Title of the section")
    description: str = Field(description="Brief description of what the section should cover")

class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(description="Description of the target audience")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")

# Define our flow state
class GuideCreatorState(BaseModel):
    topic: str = ""
    audience_level: str = ""
    guide_outline: GuideOutline = None
    sections_content: Dict[str, str] = {}

@CrewBase
class SearchCrew():
    """
    Crew specializzata nella ricerca web e analisi delle informazioni.
    Utilizza agent researcher esperto con accesso a tool di ricerca.
    """
    
    agents_config : Dict[str, Any]
    tasks_config : Dict[str, Any]
    
    @agent
    def web_researcher(self) -> Agent:
        """Agent esperto nella ricerca web con tool DuckDuckGo."""
        return Agent(
            config=self.agents_config['web_researcher'],
            tools=[CustomDuckDuckGoSearchTool()],
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=False
        )
    
    @agent
    def content_writer(self) -> Agent:
        """Agent esperto nella scrittura di contenuti educativi."""
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=False
        )
    
    @agent
    def content_reviewer(self) -> Agent:
        """Agent esperto nella revisione e miglioramento dei contenuti."""
        return Agent(
            config=self.agents_config['content_reviewer'],
            verbose=True,
            allow_delegation=False,
            max_iter=5,
            memory=False
        )
    
    @task
    def search_section_task(self) -> Task:
        """Task per la ricerca di informazioni su un argomento specifico."""
        return Task(
            config=self.tasks_config['search_section_task'],
            agent=self.web_researcher()
        )
    
    @task
    def write_section_task(self) -> Task:
        """Task per scrivere una sezione completa su un argomento."""
        return Task(
            config=self.tasks_config['write_section_task'],
            agent=self.content_writer()
        )
    
    @task
    def review_section_task(self) -> Task:
        """Task per rivedere e migliorare una sezione scritta."""
        return Task(
            config=self.tasks_config['review_section_task'],
            agent=self.content_reviewer()
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Assembla e restituisce la Crew configurata.
        
        Returns:
            Crew: La crew pronta per l'esecuzione con kickoff()
        """
        return Crew(
            agents=[self.agents_config],
            tasks=[self.tasks_config],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False,
            max_rpm=10,
            share_crew=False
        )
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Metodo di convenienza per eseguire direttamente la ricerca.
        
        Args:
            query: La query di ricerca da processare
            
        Returns:
            Dict contenente il risultato della ricerca
        """
        # Prepara gli input per tutti i task
        inputs = {
            "section_title": query,  # Compatibile con i task YAML
            "query": query  # Mantiene compatibilità con il nome originale
        }
        
        result = self.crew().kickoff(inputs=inputs)
        return {"result": result, "query": query}
    
    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """
        Esegue la crew con configurazione basata sugli input.
        
        Args:
            inputs: Dizionario contenente:
                - section_title o query: l'argomento da ricercare
                - mode (opzionale): modalità di esecuzione ('search_only', 'full', ecc.)
        
        Returns:
            Risultato dell'esecuzione della crew
        """
        # Normalizza gli input per compatibilità
        if "query" in inputs and "section_title" not in inputs:
            inputs["section_title"] = inputs["query"]
        elif "section_title" in inputs and "query" not in inputs:
            inputs["query"] = inputs["section_title"]
        
        mode = inputs.get("mode", "full")
        
        # Seleziona i task in base alla modalità
        if mode == "search_only":
            # Solo ricerca
            tasks = [self.search_section_task()]
            agents = [self.web_researcher()]
        elif mode == "write_only":
            # Solo scrittura (presume che ci siano già informazioni)
            tasks = [self.write_section_task()]
            agents = [self.content_writer()]
        elif mode == "search_and_write":
            # Ricerca e scrittura senza revisione
            tasks = [self.search_section_task(), self.write_section_task()]
            agents = [self.web_researcher(), self.content_writer()]
        else:  # mode == "full" o qualsiasi altro valore
            # Pipeline completa: ricerca, scrittura e revisione
            tasks = [
                self.search_section_task(),
                self.write_section_task(),
                self.review_section_task()
            ]
            agents = [
                self.web_researcher(),
                self.content_writer(),
                self.content_reviewer()
            ]
        
        # Crea la crew con la configurazione selezionata
        configured_crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            cache=True,
            max_rpm=10,
            share_crew=False
        )
        
        return configured_crew.kickoff(inputs=inputs)


# Test standalone della crew
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST STANDALONE - SearchCrew")
    print("="*60)
    
    # Crea un'istanza della crew
    search_crew = SearchCrew()
    
    # Query di test
    test_query = input("\nInserisci una query di test (o premi Enter per usare quella di default): ").strip()
    if not test_query:
        test_query = "Latest developments in artificial intelligence 2024"
    
    # Chiedi la modalità
    print("\nModalità disponibili:")
    print("1. search_only - Solo ricerca")
    print("2. write_only - Solo scrittura")
    print("3. search_and_write - Ricerca e scrittura")
    print("4. full - Pipeline completa (default)")
    
    mode_choice = input("\nScegli modalità (1-4, default=4): ").strip()
    mode_map = {
        "1": "search_only",
        "2": "write_only",
        "3": "search_and_write",
        "4": "full",
        "": "full"
    }
    mode = mode_map.get(mode_choice, "full")
    
    print(f"\n📍 Query di test: '{test_query}'")
    print(f"⚙️  Modalità: {mode}")
    print("-"*40)
    
    # Esegui la ricerca
    try:
        result = search_crew.kickoff({
            "query": test_query,
            "mode": mode
        })
        print("\n✅ Ricerca completata!")
        print("\n📊 RISULTATO:")
        print("-"*40)
        print(result)
    except Exception as e:
        print(f"\n❌ Errore durante il test: {e}")
