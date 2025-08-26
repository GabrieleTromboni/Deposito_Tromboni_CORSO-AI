from crewai import Flow, Agent, Crew, Task, Process
from  crewai.flow.flow import start, listen
from crewai.project import agent, task, crew
from langchain_community.tools import DuckDuckGoSearchResults
import os
from dotenv import load_dotenv

#Load variables
load_dotenv()

class InternetFlow(Flow):
    def __init__(self, topic):
        super().__init__()
        self.topic = topic if topic else "Intelligenza Artificiale 2025"
        self.agents = [self.search_agent]
        self.tasks = [self.summarize_search_results]
    
    # 1. Tool per output risultati in structured format
    def get_tool(self):
        return DuckDuckGoSearchResults()

    # 1. Define l'agente
    @agent
    def search_agent(self) -> Agent:
        # Tool di ricerca sul web
        return Agent(
            role="Esperto ricercatore di ricerche online sul web.",
            goal="Fornire un riassunto accurato e completo, con tutte le informazioni principali, dai primi 3 risultati trovati sul web up-to-date.",
            backstory=(
                "Agente AI specializzato nell'analisi di fonti web. È in grado di interpretare più risultati "
                "di ricerca, estrarre i punti principali e creare un riassunto coerente."
            ),
            tools=[self.get_tool()],
            verbose=True,
        )

    # 2. Define the Task
    @task
    def summarize_search_results(self) -> Task:
        return Task(
            description=f"""Cerca su Internet informazioni aggiornate sull'argomento '{self.topic}'. Trova almeno 3 fonti attendibili, analizza i primi 3 risultati restituiti e scrivi un riassunto chiaro e conciso.""",
            expected_output="Un riassunto chiaro e sintetico basato sui primi 3 risultati principali trovati online.",
            agent=self.search_agent,
        )

    # 3. Compose the flow
    @start()
    def take_topic(self, input=None):
        self.topic = input if input else self.topic
        print(f"\n=== Ricerca informazioni su: {self.topic} ===\n")
        return self.topic
    
    @listen(take_topic)
    def make_research(self):
        self.kickoff(self.summarize_search_results)
        print(f"\n=== Fine ricerca su: {self.topic} ===\n")

    # 4. Composizione della crew (gli agenti e i task)
    @crew
    def internet_crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

if __name__ == "__main__":
    argomento = input("Inserisci l'argomento da cercare: ")

    flow = InternetFlow(argomento)
    output = flow.kickoff()

    print("\n RISULTATO DELLA RICERCA:\n")
    print(output)