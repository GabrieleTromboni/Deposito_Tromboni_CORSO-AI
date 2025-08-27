from crewai import Agent, Crew, Task, Process
from crewai.flow.flow import Flow, start, listen
from tools.custom_tool import CustomDuckDuckGoSearchTool
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import LLM
from typing import List, Dict, Type, Optional, Any

#Load variables
load_dotenv()

# Configurazione Outline dalla Ricerca
class Section(BaseModel):
    title: str = Field(description="Title of the section")
    description: str = Field(description="Brief description of what the section should cover")

class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(description="Description of the target audience")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")


class InternetFlow(Flow):
    def __init__(self, topic=None):
        super().__init__()
        self.topic = topic if topic else "Intelligenza Artificiale 2025"
    
    # 1. Tool per output risultati in structured format
    def get_tool(self):
        return CustomDuckDuckGoSearchTool()

    # 1. Define l'agente
    def search_agent(self) -> Agent:
        # Tool di ricerca sul web
        return Agent(
            role="Esperto ricercatore di ricerche online sul web e redattore di contenuti.",
            goal="Fornire un riassunto accurato e completo, con tutte le informazioni principali, dai primi 3 risultati trovati sul web up-to-date.",
            backstory=(
                "Agente AI specializzato nell'analisi di fonti web. È in grado di interpretare più risultati "
                "di ricerca, estrarre i punti principali e creare un riassunto coerente."
            ),
            tools=[self.get_tool()],
            llm=LLM(
                model='azure/gpt-4o'
                # Rimuoviamo structured_output dall'Agent - lo useremo nel Task
            ),
            verbose=True,
            allow_delegation=False,
            max_iter=3  # Limita le iterazioni per evitare loop infiniti
        )

    # 2. Define the Task con output strutturato
    def summarize_search_results(self) -> Task:
        return Task(
            description=f"""Usa il tool DuckDuckGo Search per cercare informazioni aggiornate sull'argomento '{self.topic}'. 
            
            Esegui almeno 2-3 ricerche con query diverse per ottenere informazioni complete:
            1. Prima ricerca con la query esatta: "{self.topic}"
            2. Seconda ricerca con termini correlati o specifici
            3. Se necessario, una terza ricerca per approfondimenti
            
            Analizza tutti i risultati e crea una guida strutturata.
            
            DEVI restituire un output JSON che segua ESATTAMENTE questa struttura:
            {{
                "title": "Titolo della guida su {self.topic}",
                "introduction": "Introduzione dettagliata all'argomento basata sui risultati della ricerca",
                "target_audience": "Descrizione del pubblico target per questa guida",
                "sections": [
                    {{
                        "title": "Titolo della sezione 1",
                        "description": "Descrizione dettagliata di cosa copre questa sezione"
                    }},
                    {{
                        "title": "Titolo della sezione 2", 
                        "description": "Descrizione dettagliata di cosa copre questa sezione"
                    }},
                    {{
                        "title": "Titolo della sezione 3",
                        "description": "Descrizione dettagliata di cosa copre questa sezione"
                    }}
                ],
                "conclusion": "Conclusione o riassunto della guida basato sui risultati trovati"
            }}
            
            Basa il contenuto sui risultati di ricerca più rilevanti che trovi.
            """,
            expected_output="Un JSON strutturato con title, introduction, target_audience, sections (array di oggetti con title e description), e conclusion.",
            agent=self.search_agent(),
            output_pydantic=GuideOutline  # Usa output_pydantic invece di structured_output nel LLM
        )

    # 3. Compose the flow
    @start()
    def take_topic(self, input_data: str = None):  # Cambiato da input a input_data
        self.topic = input_data if input_data else self.topic
        print(f"\n=== Ricerca informazioni su: {self.topic} ===\n")
        return self.topic
    
    @listen(take_topic)
    def make_research(self, topic: str):  # Aggiungi il parametro topic
        crew = self.internet_crew()  # Crea la crew
        result = crew.kickoff()  # Esegui la crew
        
        # Se result è un oggetto Pydantic, convertilo in dict per visualizzarlo meglio
        if hasattr(result, 'pydantic'):
            print(f"\n=== Risultato strutturato ===")
            if isinstance(result.pydantic, GuideOutline):
                guide = result.pydantic
                print(f"Titolo: {guide.title}")
                print(f"Introduzione: {guide.introduction}")
                print(f"Target: {guide.target_audience}")
                print(f"Sezioni:")
                for section in guide.sections:
                    print(f"  - {section.title}: {section.description}")
                print(f"Conclusione: {guide.conclusion}")
        
        print(f"\n=== Fine ricerca su: {self.topic} ===\n")
        return result

    # 4. Composizione della crew (gli agenti e i task) - SENZA DECORATORE @crew
    def internet_crew(self) -> Crew:
        return Crew(
            agents=[self.search_agent()],  # Chiama il metodo per ottenere l'agente
            tasks=[self.summarize_search_results()],  # Chiama il metodo per ottenere il task
            process=Process.sequential,
            verbose=True,
        )
    
def kickoff():
    """Run the summarizer research flow"""
    argomento = input("Inserisci l'argomento da cercare: ")
    flow = InternetFlow(argomento)
    flow.kickoff(inputs={"input_data": argomento})  # Passa l'input correttamente
    print("\n=== Flow Complete ===")
    print("Your summarizer research results are ready.")

def plot():
    """Generate a visualization of the flow"""
    flow = InternetFlow()
    flow.plot("internet_flow")
    print("Flow visualization saved to internet_flow.html")

if __name__ == "__main__":
    kickoff()
    plot()