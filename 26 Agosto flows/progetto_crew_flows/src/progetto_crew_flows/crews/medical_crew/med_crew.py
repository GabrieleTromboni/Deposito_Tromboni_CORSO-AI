from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import tool
from langchain_community.vectorstores import FAISS
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass


load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "26 Agosto flows/progetto_crew_flows/faiss_index_example"
    # Text splitting
    chunk_size: int = 500
    chunk_overlap: int = 80
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                     # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # nome del modello in LM Studio, via env var


SETTINGS = Settings()
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

@tool("medical_rag_search")
def search_medical_database(query: str, topic: str) -> str:
    """
    Search the medical vector database for relevant information
    """
    from langchain_openai import AzureOpenAIEmbeddings
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=API_KEY,
        azure_endpoint=CLIENT_AZURE,
        model=EMBEDDING_MODEL
    )
    
    # Load FAISS index
    persist_dir = "25 Agosto/faiss_index_example"
    if not Path(persist_dir).exists():
        return "Vector database not found. Please ensure RAG database is properly initialized."
    
    vector_store = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Perform similarity search
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    docs = retriever.invoke(f"{topic}: {query}")
    
    # Format results with sources
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        results.append(f"[Source: {source}]\n{content}")
    
    return "\n\n---\n\n".join(results)

@CrewBase
class MedicalCrew:
    """Medical crew for RAG-based information retrieval"""
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    def __init__(self):
        self.medical_search_tool = search_medical_database
    
    @agent
    def medical_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['medical_researcher'],
            tools=[self.medical_search_tool],
            verbose=True
        )
    
    @agent
    def medical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['medical_analyst'],
            verbose=True
        )
    
    @task
    def retrieve_medical_info(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_medical_info'],
            tools=[self.medical_search_tool]
        )
    
    @task
    def synthesize_medical_response(self) -> Task:
        return Task(
            config=self.tasks_config['synthesize_medical_response'],
            output_pydantic=GuideOutlineResult
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process="sequential",
            verbose=True
        )

class GuideOutlineResult:
    """Output format for medical responses"""
    def __init__(self, content: str, sources: List[str], sections: List[Dict] = None):
        self.raw = content
        self.sources = sources
        self.sections = sections or []
