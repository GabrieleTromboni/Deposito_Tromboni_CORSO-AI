from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, List

import os
import sys
from pathlib import Path
# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.rag_tool import generate_documents, create_vectordb, store_in_vectordb

@CrewBase
class DatabaseCrew():
    '''
    Crew to create the RAG Vector Database.
    It includes:
        - Agents which generate documents and store them in the vector database.
        - Tools to generate and manage documents.
    '''
    
    agents_config : List[BaseAgent]
    tasks_config : List[Task]

    def __init__(self):

        # Ensure database directory exists
        self.db_path = Path("rag_database")
        self.db_path.mkdir(exist_ok=True)
        
    @agent
    def document_generator(self) -> Agent:
        '''Agent to generate domain-related documents.'''
        return Agent(
            config=self.agents_config['document_generator'],
            tools=[generate_documents],
            verbose=True
        )
    
    @agent
    def database_engineer(self) -> Agent:
        '''Agent to manage the vector database.'''
        return Agent(
            config=self.agents_config['database_engineer'],
            tools=[create_vectordb, store_in_vectordb],
            verbose=True
        )
    
    @task
    def generate_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_documents_task']
        )
    
    @task
    def create_rag_database_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_database_task']
        )

    @task
    def store_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['store_documents_task'],
            context=[
                self.generate_documents_task,
                self.create_rag_database_task
            ]
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.document_generator, self.database_engineer],
            tasks=[self.generate_documents_task, self.create_rag_database_task, self.store_documents_task],
            process=Process.sequential,
            verbose=True
        )

    def kickoff(self, subjects: Dict[str, List[str]]) -> bool:
        '''Start the crew process for database initialization.'''
        return self.crew.kickoff(subjects)