"""
Crews package for the progetto_crew_flows project.

This package contains all the CrewAI crew implementations:
- RAGCrew: Original RAG crew for basic operations
- DatabaseCrew: New unified crew for database operations and RAG retrieval
"""

from .rag_crew.rag_crew import RAGCrew
from .database_crew.database_crew import DatabaseCrew, create_database_crew

__all__ = [
    'RAGCrew',
    'DatabaseCrew', 
    'create_database_crew'
]
