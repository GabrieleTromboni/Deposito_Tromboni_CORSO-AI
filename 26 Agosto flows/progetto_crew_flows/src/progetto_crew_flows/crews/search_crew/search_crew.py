"""
SearchCrew - Crew specializzata per la ricerca web.
Utilizza agent researcher con tool di ricerca DuckDuckGo personalizzato.
"""

"""
SearchCrew - Crew specializzata per la ricerca web.
Utilizza agent researcher con tool di ricerca DuckDuckGo personalizzato.
"""

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, Any, List
import sys
import os

# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.search_tool import CustomDuckDuckGoSearchTool

@CrewBase
class SearchCrew():
    """
    Crew specializzata nella ricerca web e analisi delle informazioni.
    Utilizza agent researcher esperto con accesso a tool di ricerca.
    """
    
    agents_config : List[BaseAgent]
    tasks_config : List[Task]
    
    @agent
    def web_researcher(self) -> Agent:
        """Agent esperto nella ricerca web con tool DuckDuckGo."""
        return Agent(
            config=self.agents_config['web_researcher'],
            tools=[CustomDuckDuckGoSearchTool()],
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            memory=False  # Disabilita memoria per evitare ChromaDB
        )
    
    @agent
    def content_writer(self) -> Agent:
        """Agent esperto nella scrittura di contenuti educativi."""
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True,
            allow_delegation=False,
            max_iter=2,
            memory=False  # Disabilita memoria per evitare ChromaDB
        )
    
    @agent
    def content_reviewer(self) -> Agent:
        """Agent esperto nella revisione e miglioramento dei contenuti."""
        return Agent(
            config=self.agents_config['content_reviewer'],
            verbose=True,
            allow_delegation=False,
            max_iter=2,
            memory=False  # Disabilita memoria per evitare ChromaDB
        )
    
    @task
    def search_section_task(self) -> Task:
        """Task per la ricerca di informazioni su un argomento specifico."""
        return Task(
            config=self.tasks_config['search_section_task']
        )
    
    @task
    def write_section_task(self) -> Task:
        """Task per scrivere una sezione completa su un argomento."""
        return Task(
            config=self.tasks_config['write_section_task'],
            context=[self.search_section_task()]
        )
    
    @task
    def review_section_task(self) -> Task:
        """Task per rivedere e migliorare una sezione scritta."""
        return Task(
            config=self.tasks_config['review_section_task'],
            context=[self.write_section_task()]
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Assembla e restituisce la Crew configurata.
        
        Returns:
            Crew: La crew pronta per l'esecuzione con kickoff()
        """
        return Crew(
            agents=[
                self.web_researcher(),
                self.content_writer(), 
                self.content_reviewer()
            ],
            tasks=[
                self.search_section_task(),
                self.write_section_task(),
                self.review_section_task()
            ],
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabilita memoria per evitare ChromaDB
            cache=False,   # Disabilita cache per evitare problemi
            embedder=None, # Disabilita embedder esplicitamente
            max_rpm=10,
            share_crew=False
        )