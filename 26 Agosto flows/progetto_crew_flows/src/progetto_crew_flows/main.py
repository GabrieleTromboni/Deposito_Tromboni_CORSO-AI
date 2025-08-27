from crewai import Agent, Crew, Task, Process
from  crewai.flow.flow import Flow, start, listen
from crewai.project import agent, task, crew
from crewai_tools import SerperDevTool
import os
import json
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from crewai import LLM
from progetto_crew_flows.crews.content_crew.content_crew import ContentCrew
from typing import List, Dict

#Load variables
load_dotenv()

# Configurazione Azure OpenAI LLM - PARAMETRI CORRETTI
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
    def __init__(self, topic):
        super().__init__()
        self.topic = topic if topic else "Intelligenza Artificiale 2025"
        self.agents = [self.search_agent]
        self.tasks = [self.summarize_search_results]
    
    # 1. Tool per output risultati in structured format
    def get_tool(self):
        return SerperDevTool()

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
            llm=LLM(model='azure/gpt-4o',
                    response_format=GuideOutline),
            verbose=True,
            allow_delegation=False,
            max_iter=3  # Limita le iterazioni per evitare loop infiniti
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
    
def kickoff():
    """Run the summarizer research flow"""
    argomento = input("Inserisci l'argomento da cercare: ")
    InternetFlow(argomento).kickoff()
    print("\n=== Flow Complete ===")
    print("Your summarizer research results are ready.")
    print("Open output/complete_guide.md to view it.")

def plot():
    """Generate a visualization of the flow"""
    flow = InternetFlow()
    flow.plot("internet_flow")
    print("Flow visualization saved to internet_flow.html")

if __name__ == "__main__":
    kickoff()