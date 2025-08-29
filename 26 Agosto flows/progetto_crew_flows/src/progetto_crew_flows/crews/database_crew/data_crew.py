from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from typing import Dict, List
import os
import sys
from pathlib import Path
import os

# Fix the import path for tools
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.rag_tool import generate_documents, create_vectordb, store_in_vectordb
except ImportError as e:
    print(f"Warning: Could not import RAG tools: {e}")
    # Create dummy functions if tools are not available
    def generate_documents(*args, **kwargs):
        return "Dummy document generation"
    def create_vectordb(*args, **kwargs):
        return "Dummy vectordb creation"
    def store_in_vectordb(*args, **kwargs):
        return "Dummy storage"

@CrewBase
class DatabaseCrew():
    '''
    Crew to create the RAG Vector Database.
    It includes:
        - Agents which generate documents and store them in the vector database.
        - Tools to generate and manage documents.
    '''
    
    # Initialize config files path
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Ensure database directory exists, independent from CWD
        project_root = Path(__file__).resolve().parents[3]
        db_dir = os.getenv("RAG_DB_DIR") or str(project_root / "RAG_database")
        self.db_path = Path(db_dir)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
    @agent
    def document_generator(self) -> Agent:
        '''Agent to generate domain-related documents.'''
        return Agent(
            config=self.agents_config['document_generator'],
            tools=[generate_documents],
            verbose=True,
            allow_delegation=False
        )
    
    @agent
    def database_engineer(self) -> Agent:
        '''Agent to manage the vector database.'''
        return Agent(
            config=self.agents_config['database_engineer'],
            tools=[create_vectordb, store_in_vectordb],
            verbose=True,
            allow_delegation=False
        )
    
    @task
    def generate_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['generation_documents_task'],
            agent=self.document_generator()
        )
    
    @task
    def create_rag_database_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_database_task'],
            agent=self.database_engineer()
        )

    @task
    def store_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['store_documents_task'],
            agent=self.database_engineer(),
            context=[
                self.generate_documents_task(),
                self.create_rag_database_task()
            ]
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the database crew"""
        return Crew(
            agents=[
                self.document_generator(),
                self.database_engineer()
            ],
            tasks=[
                self.generate_documents_task(),
                self.create_rag_database_task(),
                self.store_documents_task()
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def kickoff(self, subjects: Dict[str, List[str]] = None):
        """Initialize database with all subjects and topics"""
        if subjects is None:
            subjects = {}
            
        # Flatten all topics from subjects
        all_topics = []
        for subject, topics in subjects.items():
            for topic in topics:
                all_topics.append(f"{subject} - {topic}")
        
        # Provide a default topic if none given
        if not all_topics:
            all_topics = ["General Knowledge"]
        
        # Run the crew with proper input format - ensure topics is a list
        inputs = {
            "topic": ", ".join(all_topics),
            "topics": all_topics,  # Pass as list for generate_documents tool
            "subjects": subjects,
            "db_name": str(self.db_path)  # Add database path
        }
        
        try:
            result = self.crew().kickoff(inputs=inputs)
            return result
        except Exception as e:
            print(f"Error during crew execution: {e}")
            # Return a default result structure
            return {
                "status": "error",
                "message": str(e),
                "topics_processed": all_topics
            }