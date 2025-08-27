"""
SearchCrew - Crew specializzata per la ricerca web.
Utilizza un agent researcher con tool di ricerca DuckDuckGo personalizzato.
"""

from crewai import Agent, Task, Crew, Process
from typing import Dict, Any
import sys
import os

# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.custom_tool import CustomDuckDuckGoSearchTool


class SearchCrew:
    """
    Crew specializzata nella ricerca web e analisi delle informazioni.
    Utilizza un agent researcher esperto con accesso a tool di ricerca.
    """
    
    def __init__(self):
        """Inizializza la SearchCrew con agent e task configurati."""
        self.search_tool = CustomDuckDuckGoSearchTool()
        self._setup_agents()
        self._setup_tasks()
    
    def _setup_agents(self):
        """
        Configura gli agent della crew.
        In questo caso, un singolo agent researcher.
        """
        self.researcher = Agent(
            role="Senior Web Researcher",
            goal="Find comprehensive and accurate information from the web about any topic",
            backstory="""You are an expert web researcher with years of experience in 
            finding, analyzing, and synthesizing information from various online sources. 
            You excel at understanding user queries and providing relevant, well-structured 
            information. You always verify facts and provide balanced, objective summaries.""",
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            tools=[self.search_tool],
            memory=True
        )
    
    def _setup_tasks(self):
        """
        Configura i task della crew.
        Definisce come l'agent deve processare le query di ricerca.
        """
        self.search_task = Task(
            description="""
            Conduct a comprehensive web search for the following query: {query}
            
            Your search should:
            1. Find the most relevant and recent information
            2. Identify key facts and important details
            3. Organize the information in a clear, structured manner
            4. Provide a balanced view if the topic is controversial
            5. Include relevant statistics or data if available
            
            Focus on accuracy and relevance over quantity.
            """,
            expected_output="""
            A well-structured report containing:
            - Executive Summary (2-3 sentences)
            - Key Findings (bullet points)
            - Detailed Information (organized by subtopics)
            - Sources consulted
            - Recommendations or conclusions if applicable
            """,
            agent=self.researcher,
            output_file=None  # Non salva su file, restituisce solo il risultato
        )
    
    def crew(self) -> Crew:
        """
        Assembla e restituisce la Crew configurata.
        
        Returns:
            Crew: La crew pronta per l'esecuzione con kickoff()
        """
        return Crew(
            agents=[self.researcher],
            tasks=[self.search_task],
            process=Process.sequential,  # Processo sequenziale (un solo task)
            verbose=True,
            memory=True,
            cache=True,
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
        crew = self.crew()
        result = crew.kickoff(inputs={"query": query})
        return {"result": result, "query": query}


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
    
    print(f"\n📍 Query di test: '{test_query}'")
    print("-"*40)
    
    # Esegui la ricerca
    try:
        result = search_crew.execute(test_query)
        print("\n✅ Ricerca completata!")
        print("\n📊 RISULTATO:")
        print("-"*40)
        print(result.get("result"))
    except Exception as e:
        print(f"\n❌ Errore durante il test: {e}")
