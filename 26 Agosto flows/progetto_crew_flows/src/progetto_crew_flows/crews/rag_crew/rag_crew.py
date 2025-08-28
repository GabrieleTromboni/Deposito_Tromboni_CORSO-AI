from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
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

@tool("generate_medical_documents")
def generate_medical_documents(topic: str) -> str:
    """Generate medical documents for a specific topic"""
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = init_chat_model(
        model=CHAT_MODEL,
        api_version="2024-02-15-preview",
        azure_endpoint=CLIENT_AZURE,
        api_key=API_KEY,
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical content generator. Create detailed medical information."),
        ("human", "Generate comprehensive medical content about {topic}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"topic": topic})
    
    return result.content if hasattr(result, 'content') else str(result)

@tool("store_in_vectordb")
def store_in_vectordb(content: str, topic: str) -> str:
    """Store generated content in FAISS vector database"""
    embeddings = AzureOpenAIEmbeddings(
        api_key=API_KEY,
        azure_endpoint=CLIENT_AZURE,
        model=EMBEDDING_MODEL
    )
    
    # Create document
    doc = Document(
        page_content=content,
        metadata={"topic": topic, "source": f"generated_{topic}.md"}
    )
    
    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap
    )
    chunks = splitter.split_documents([doc])
    
    # Load or create vector store
    persist_dir = SETTINGS.persist_dir
    if Path(persist_dir).exists():
        vector_store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(persist_dir)
    
    return f"Successfully stored {len(chunks)} chunks for topic: {topic}"

@CrewBase
class RagCrew():
    '''
    Crew to create the RAG Vector Database.
    It includes:
        - Agents which generate documents and store them in the vector database.
        - Tools to retrieve and manage documents from the vector database.
    '''
    
    @agent
    def document_generator(self) -> Agent:
        return Agent(
            role="Medical Document Generator",
            goal="Generate comprehensive medical documents for various topics",
            backstory="You are an expert in medical content creation with deep knowledge across specialties.",
            tools=[generate_medical_documents],
            verbose=True
        )
    
    @agent
    def database_manager(self) -> Agent:
        return Agent(
            role="Vector Database Manager",
            goal="Store and organize documents in the FAISS vector database",
            backstory="You manage the vector database ensuring efficient storage and retrieval.",
            tools=[store_in_vectordb],
            verbose=True
        )
    
    @task
    def generate_documents_task(self) -> Task:
        return Task(
            description="Generate medical documents for topic: {topic}",
            expected_output="Comprehensive medical content",
            agent=self.document_generator
        )
    
    @task
    def store_documents_task(self) -> Task:
        return Task(
            description="Store generated documents in vector database",
            expected_output="Confirmation of successful storage",
            agent=self.database_manager
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.document_generator, self.database_manager],
            tasks=[self.generate_documents_task, self.store_documents_task],
            process="sequential",
            verbose=True
        )
