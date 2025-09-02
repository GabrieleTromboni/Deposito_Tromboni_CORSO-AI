from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from typing import Dict, List
import os
import sys
from pathlib import Path

# Fix the import path for tools
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.rag_tool import generate_documents_as_files, create_vectordb, store_individual_documents, recreate_collection_for_rag, store_documents_in_qdrant
except ImportError as e:
    print(f"Warning: Could not import RAG tools v2: {e}")
    # Try fallback to original tools
    try:
        from tools.rag_tool import generate_documents, create_vectordb, store_in_vectordb
        # Create aliases for compatibility
        generate_documents_as_files = generate_documents
        store_individual_documents = store_in_vectordb
        recreate_collection_for_rag = None
        store_documents_in_qdrant = None
    except ImportError as e2:
        print(f"Warning: Could not import any RAG tools: {e2}")
        # Create dummy functions if tools are not available
        def generate_documents_as_files(*args, **kwargs):
            return "Dummy document generation"
        def create_vectordb(*args, **kwargs):
            return "Dummy vectordb creation"
        def store_individual_documents(*args, **kwargs):
            return "Dummy storage"
        def recreate_collection_for_rag(*args, **kwargs):
            return "Dummy Qdrant collection creation"
        def store_documents_in_qdrant(*args, **kwargs):
            return "Dummy Qdrant storage"

@CrewBase
class DatabaseCrew():
    '''
    Crew to create the RAG Vector Database.
    It includes:
        - Agents which generate documents and store them in the vector database.
        - Tools to generate and manage documents.
        - Support for both FAISS and Qdrant vector databases.
    '''
    
    # Initialize config files path
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, use_qdrant: bool = False):
        # Ensure database directory exists, independent from CWD
        project_root = Path(__file__).resolve().parents[3]
        db_dir = os.getenv("RAG_DB_DIR") or str(project_root / "RAG_database")
        self.db_path = Path(db_dir)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Set vector database type
        self.use_qdrant = use_qdrant
        print(f"💾 DatabaseCrew initialized with {'Qdrant' if use_qdrant else 'FAISS'} vector database")
        
        if use_qdrant:
            # Validate Qdrant configuration
            if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_COLLECTION"):
                print("⚠️ Warning: Qdrant configuration missing. Set QDRANT_URL and QDRANT_COLLECTION environment variables.")
                print("   Falling back to FAISS...")
                self.use_qdrant = False
        
    @agent
    def document_generator(self) -> Agent:
        '''Agent to generate domain-related documents as individual files.'''
        return Agent(
            config=self.agents_config['document_generator'],
            tools=[generate_documents_as_files],
            verbose=True,
            allow_delegation=False
        )
    
    @agent
    def database_engineer(self) -> Agent:
        '''Agent to manage the vector database (FAISS or Qdrant).'''
        if self.use_qdrant:
            tools = [recreate_collection_for_rag, store_documents_in_qdrant]
        else:
            tools = [create_vectordb, store_individual_documents]
            
        return Agent(
            config=self.agents_config['database_engineer'],
            tools=tools,
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
        '''Task to create/recreate the vector database (FAISS or Qdrant).'''
        return Task(
            config=self.tasks_config['create_database_task'],
            agent=self.database_engineer()
        )

    @task
    def store_documents_task(self) -> Task:
        '''Task to store documents in the vector database (FAISS or Qdrant).'''
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
    
    def kickoff(self, subjects: Dict[str, List[str]] = None, docs_per_topic: int = 1, max_tokens_per_doc: int = 600, batch_size: int = 3):
        """Initialize database with all subjects and topics with optimized document generation and rate limiting
        
        Args:
            subjects: Dictionary of subjects with their topics
            docs_per_topic: Number of documents to generate per topic (default: 1)
            max_tokens_per_doc: Maximum tokens per document for efficiency (default: 800)
            batch_size: Number of topics to process in each batch to avoid rate limits (default: 3)
        """
        if subjects is None:
            subjects = {}
        
        # Check if database already exists and has content
        if self.use_qdrant:
            # For Qdrant, we would need to check collection status
            # For now, just proceed with creation/recreation
            print(f"🔧 Using Qdrant vector database")
        else:
            # Check FAISS database
            db_index_path = self.db_path / "index.faiss"
            if db_index_path.exists():
                print(f"⚠️ FAISS RAG database already exists at {self.db_path}")
                print(f"   Skipping database initialization to avoid duplicates.")
                return {
                    "status": "skipped",
                    "message": "FAISS Database already exists",
                    "db_path": str(self.db_path),
                    "db_type": "FAISS"
                }
            
        # Flatten all topics from subjects
        all_topics = []
        for subject, topics in subjects.items():
            for topic in topics:
                all_topics.append(f"{subject} - {topic}")
        
        # Provide a default topic if none given
        if not all_topics:
            all_topics = ["General Knowledge"]
        
        total_batches = (len(all_topics) + batch_size - 1) // batch_size
        estimated_time = total_batches * 2.5  # Rough estimate including delays
        
        db_type = "Qdrant" if self.use_qdrant else "FAISS"
        print(f"🚀 Optimized Document Generation Strategy ({db_type}):")
        print(f"   • {docs_per_topic} documents per topic")
        print(f"   • Max {max_tokens_per_doc} tokens per document")
        print(f"   • {len(all_topics)} topics in {total_batches} batches of {batch_size}")
        print(f"   • Expected ~{len(all_topics) * docs_per_topic * max_tokens_per_doc:,} total tokens")
        print(f"   • Estimated time: ~{estimated_time:.0f}s with rate limiting")
        print(f"   • Diverse content strategies for better RAG performance")
        print(f"   • Database type: {db_type}")
        
        # Run the crew with proper input format - ensure topics is a list
        inputs = {
            "topic": ", ".join(all_topics),
            "topics": all_topics,  # Pass as list for generate_documents tool
            "docs_per_topic": docs_per_topic,  # Add docs_per_topic parameter
            "max_tokens_per_doc": max_tokens_per_doc,  # Add token limit parameter
            "batch_size": batch_size,  # Add batch size for rate limiting
            "subjects": subjects,
            "db_name": str(self.db_path),  # Add database path
            "total_topics": len(all_topics),  # Add total count for validation
            "use_qdrant": self.use_qdrant,  # Add database type flag
            "db_type": db_type  # Add human-readable database type
        }
        
        try:
            print(f"\n🔄 Starting {db_type} database crew execution...")
            result = self.crew().kickoff(inputs=inputs)
            print(f"✅ {db_type} database crew execution completed!")
            return result
        except Exception as e:
            print(f"❌ Error during {db_type} crew execution: {e}")
            import traceback
            traceback.print_exc()
            # Return a default result structure
            return {
                "status": "error",
                "message": str(e),
                "topics_processed": all_topics,
                "db_type": db_type
            }