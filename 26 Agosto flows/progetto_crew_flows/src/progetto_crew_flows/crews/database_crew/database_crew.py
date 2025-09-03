"""
Database Crew for managing vector database operations and RAG retrieval.

This crew handles both FAISS and Qdrant database operations, including:
- Database creation and population
- Intelligent RAG retrieval with multiple search strategies
- Content formatting and guide generation
"""

from typing import Dict, Any, Optional, List
import json
from pathlib import Path

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, crew, task

from ...tools.rag_tool import (
    # Qdrant tools
    recreate_collection_for_rag,
    store_documents_in_qdrant, 
    intelligent_rag_search,
    qdrant_hybrid_search,
    qdrant_semantic_search,
    qdrant_text_search,
    
    # FAISS tools
    create_vectordb,
    store_individual_documents,
    retrieve_from_vectordb,
    
    # Formatting tools
    format_content_as_guide
)

@CrewBase
class DatabaseCrew:
    """Database operations crew for both FAISS and Qdrant databases"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self):
        self.database_type = "faiss"  # Default to FAISS
        self.available_databases = []
        
    def set_database_type(self, db_type: str):
        """Set the database type for operations"""
        self.database_type = db_type.lower()
        
    def set_available_databases(self, databases: List[str]):
        """Set list of available databases"""
        self.available_databases = databases

    @agent
    def database_manager(self) -> Agent:
        """Database operations manager agent"""
        tools = [create_vectordb, store_individual_documents, retrieve_from_vectordb]
        if "qdrant" in self.available_databases:
            tools.extend([recreate_collection_for_rag, store_documents_in_qdrant])
        
        return Agent(
            config=self.agents_config['database_manager'],
            tools=tools,
            verbose=True
        )

    @agent 
    def qdrant_specialist(self) -> Agent:
        """Qdrant database specialist agent"""
        return Agent(
            config=self.agents_config['qdrant_specialist'],
            tools=[
                recreate_collection_for_rag,
                store_documents_in_qdrant,
                intelligent_rag_search,
                qdrant_hybrid_search,
                qdrant_semantic_search,
                qdrant_text_search
            ],
            verbose=True
        )

    @agent
    def rag_retrieval_specialist(self) -> Agent:
        """RAG retrieval specialist agent"""
        tools = [retrieve_from_vectordb]  # FAISS tools
        
        if "qdrant" in self.available_databases:
            tools.extend([
                intelligent_rag_search,
                qdrant_hybrid_search, 
                qdrant_semantic_search,
                qdrant_text_search
            ])
        
        return Agent(
            config=self.agents_config['rag_retrieval_specialist'],
            tools=tools,
            verbose=True
        )

    @agent
    def content_formatter(self) -> Agent:
        """Content formatting specialist agent"""
        return Agent(
            config=self.agents_config['content_formatter'],
            tools=[format_content_as_guide],
            verbose=True
        )

    @task
    def create_qdrant_database_task(self) -> Task:
        """Task for creating Qdrant database"""
        return Task(
            config=self.tasks_config['create_qdrant_database_task'],
            agent=self.qdrant_specialist()
        )

    @task
    def create_faiss_database_task(self) -> Task:
        """Task for creating FAISS database"""
        return Task(
            config=self.tasks_config['create_faiss_database_task'],
            agent=self.database_manager()
        )

    @task
    def execute_rag_retrieval_task(self) -> Task:
        """Task for executing RAG retrieval"""
        return Task(
            config=self.tasks_config['execute_rag_retrieval_task'],
            agent=self.rag_retrieval_specialist()
        )

    @task
    def format_rag_results_task(self) -> Task:
        """Task for formatting RAG results"""
        return Task(
            config=self.tasks_config['format_rag_results_task'],
            agent=self.content_formatter()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Database Crew"""
        # Determine tasks based on operation type
        if hasattr(self, '_operation_type'):
            if self._operation_type == 'create_database':
                if self.database_type == 'qdrant':
                    tasks = [self.create_qdrant_database_task()]
                else:
                    tasks = [self.create_faiss_database_task()]
            elif self._operation_type == 'rag_retrieval':
                tasks = [
                    self.execute_rag_retrieval_task(),
                    self.format_rag_results_task()
                ]
            else:
                # Default to RAG retrieval
                tasks = [
                    self.execute_rag_retrieval_task(),
                    self.format_rag_results_task()
                ]
        else:
            # Default to RAG retrieval
            tasks = [
                self.execute_rag_retrieval_task(),
                self.format_rag_results_task()
            ]

        return Crew(
            agents=self.agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

    def create_database(
        self,
        subject: str,
        topic: str,
        database_type: str = "faiss",
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a vector database (FAISS or Qdrant)
        
        Args:
            subject: Subject domain (e.g., 'medicine', 'football')
            topic: Specific topic (e.g., 'cardiology', 'premier league')  
            database_type: Type of database ('faiss' or 'qdrant')
            collection_name: Name for Qdrant collection (if applicable)
            
        Returns:
            Dict with database creation results
        """
        self.set_database_type(database_type)
        self._operation_type = 'create_database'
        
        inputs = {
            'subject': subject,
            'topic': topic,
            'database_type': database_type,
            'collection_name': collection_name or f"{subject}_{topic}"
        }
        
        result = self.crew().kickoff(inputs=inputs)
        
        try:
            if hasattr(result, 'raw'):
                return json.loads(result.raw)
            elif isinstance(result, str):
                return json.loads(result)
            else:
                return {'status': 'success', 'result': str(result)}
        except json.JSONDecodeError:
            return {'status': 'success', 'result': str(result)}

    def execute_rag(
        self,
        query: str,
        subject: str,
        topic: str,
        database_type: str = "faiss",
        available_databases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG retrieval and formatting
        
        Args:
            query: User query for retrieval
            subject: Subject domain
            topic: Specific topic
            database_type: Preferred database type
            available_databases: List of available database types
            
        Returns:
            Dict with formatted guide results
        """
        self.set_database_type(database_type)
        if available_databases:
            self.set_available_databases(available_databases)
        
        self._operation_type = 'rag_retrieval'
        
        inputs = {
            'query': query,
            'subject': subject,
            'topic': topic,
            'database_type': database_type,
            'available_databases': available_databases or [database_type]
        }
        
        result = self.crew().kickoff(inputs=inputs)
        
        try:
            if hasattr(result, 'raw'):
                return json.loads(result.raw)
            elif isinstance(result, str):
                return json.loads(result)
            else:
                return {'status': 'success', 'result': str(result)}
        except json.JSONDecodeError:
            return {'status': 'success', 'result': str(result)}


def create_database_crew() -> DatabaseCrew:
    """Factory function to create a DatabaseCrew instance"""
    return DatabaseCrew()
