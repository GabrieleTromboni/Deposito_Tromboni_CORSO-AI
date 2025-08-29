from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from typing import List
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Union, List, Optional
from progetto_crew_flows.models import GuideOutline, Section
import json

load_dotenv()

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = "26 Agosto flows/progetto_crew_flows/RAG_database"
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

# RAG Tools
# Tool per la generazione di documenti e store in vector database

@tool
def generate_documents(topics: List[str]) -> List[Dict[str, str]]:
    """Generate documents for specific topics and return them as structured data for DatabaseCrew"""

    llm = init_chat_model(
        model=CHAT_MODEL,
        model_provider="azure_openai",
        api_version="2024-02-15-preview",
        azure_endpoint=CLIENT_AZURE,
        api_key=API_KEY,
        temperature=0.1
        )

    documents = []
    
    for topic in topics:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert {topic} content generator. Create detailed, accurate information about it."),
            ("human", f"Generate comprehensive content about {topic} including key concepts, applications, and recent developments.")
        ])
        
        chain = prompt | llm
        result = chain.invoke({"topic": topic})
        
        content = result.content if hasattr(result, 'content') else str(result)
        documents.append({
            "topic": topic,
            "content": content,
            "source": f"generated_{topic}"
        })
    
    return documents

@tool
def create_vectordb() -> str:
    '''Create or initialize the Vector Database to be used for RAG with WebRAGFlow.'''
    
    embeddings = AzureOpenAIEmbeddings(
        model_provider="azure_openai",
        api_key=API_KEY,
        azure_endpoint=CLIENT_AZURE,
        deployment=EMBEDDING_MODEL,
        api_version="2024-02-15-preview"
    )
    
    # Create directory if it doesn't exist
    Path(SETTINGS.persist_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize with empty documents
    initial_doc = Document(
        page_content="Vector database initialized",
        metadata={"source": "system", "topic": "initialization"}
    )
    
    vector_store = FAISS.from_documents([initial_doc], embeddings)
    vector_store.save_local(SETTINGS.persist_dir)

    return f"Vector database created successfully at {SETTINGS.persist_dir}"

@tool
def store_in_vectordb(
    content: Union[str, List[Dict[str, str]]],
    topic: Optional[str] = None
    ) -> str:
    """Store generated content in the FAISS vector database for use with WebRAGFlow"""
    
    embeddings = AzureOpenAIEmbeddings(
        model_provider="azure_openai",
        api_key=API_KEY,
        azure_endpoint=CLIENT_AZURE,
        deployment=EMBEDDING_MODEL,
        api_version="2024-02-15-preview"
    )
    
    # Handle different input formats
    documents = []
    if isinstance(content, list):
        for item in content:
            doc = Document(
                page_content=item.get("content", ""),
                metadata={
                    "topic": item.get("topic", "unknown"),
                    "source": item.get("source", "generated")
                }
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=content,
            metadata={"topic": topic or "general", "source": f"generated_{topic or 'content'}.md"}
        )
        documents.append(doc)
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    # Load or create vector store
    persist_dir = SETTINGS.persist_dir
    if Path(persist_dir).exists() and Path(persist_dir, "index.faiss").exists():
        vector_store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    vector_store.save_local(persist_dir)
    
    topics_stored = set([doc.metadata.get("topic", "unknown") for doc in documents])
    return f"Successfully stored {len(chunks)} chunks for topics: {', '.join(topics_stored)}"

@tool
def retrieve_from_vectordb(query: str, topic: Optional[str] = None, subject: Optional[str] = None, k: Optional[int] = None) -> str:
    """
    Retrieve raw information from the vector database without formatting.
    Returns raw documents for further processing.
    """
    
    # Initialize embeddings
    embeddings = AzureOpenAIEmbeddings(
        model_provider="azure_openai",
        api_key=API_KEY,
        azure_endpoint=CLIENT_AZURE,
        deployment=EMBEDDING_MODEL,
        api_version="2024-02-15-preview"
    )
    
    # Load FAISS index from correct path
    persist_dir = SETTINGS.persist_dir
    if not Path(persist_dir).exists() or not Path(persist_dir, "index.faiss").exists():
        return json.dumps([{"error": "Vector database not found", "content": "Please ensure RAG database is properly initialized."}])
    
    vector_store = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Build search query with both subject and topic if available
    if subject and topic:
        search_query = f"{subject} {topic}: {query}"
    elif topic:
        search_query = f"{topic}: {query}"
    else:
        search_query = query
    
    # Use custom k if provided (default to 10 for comprehensive coverage)
    search_k = k if k is not None else 10
    
    # Perform similarity search with MMR
    retriever = vector_store.as_retriever(
        search_type=SETTINGS.search_type,
        search_kwargs={
            "k": search_k,
            "fetch_k": SETTINGS.fetch_k,
            "lambda_mult": SETTINGS.mmr_lambda
        }
    )
    
    docs = retriever.invoke(search_query)
    
    # Return raw content for processing
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        doc_topic = doc.metadata.get("topic", "general")
        doc_subject = doc.metadata.get("subject", "general")
        content = doc.page_content
        results.append({
            "content": content,
            "topic": doc_topic,
            "subject": doc_subject,
            "source": source
        })
    
    if not results:
        return json.dumps([{"error": "No results", "content": f"No relevant information found for query: {search_query}"}])
    
    # Return as JSON string for easy parsing
    return json.dumps(results)

@tool
def format_content_as_guide(retrieved_info: str, query: str, topic: str, subject: Optional[str] = None) -> str:
    """
    Format retrieved content into a structured guide following GuideOutline model.
    This tool is used by content_synthesizer to create properly formatted guides.
    Returns a JSON string that can be parsed into GuideOutline.
    """
    
    llm = init_chat_model(
        model=CHAT_MODEL,
        model_provider="azure_openai",
        api_version="2024-02-15-preview",
        azure_endpoint=CLIENT_AZURE,
        api_key=API_KEY,
        temperature=0.3
    )
    
    # Parse retrieved information
    try:
        docs = json.loads(retrieved_info) if isinstance(retrieved_info, str) else retrieved_info
        # Check for errors in retrieved data
        if docs and isinstance(docs[0], dict) and "error" in docs[0]:
            # Create minimal guide with error info
            error_guide = GuideOutline(
                title=f"Limited Information Available for {topic}",
                introduction=f"Limited information is available about {topic}. {docs[0].get('content', '')}",
                target_audience="General audience",
                sections=[
                    Section(
                        title="Information Status",
                        description=docs[0].get('content', 'No information available')
                    )
                ],
                conclusion="Please check back later or consult alternative sources."
            )
            return error_guide.model_dump_json(indent=2)
    except Exception as e:
        docs = [{"content": retrieved_info, "topic": topic, "source": "unknown"}]
    
    # Combine all content
    combined_content = "\n\n".join([doc.get("content", "") for doc in docs if "error" not in doc])
    
    if not combined_content:
        # No valid content found
        minimal_guide = GuideOutline(
            title=f"Guide to {topic}",
            introduction=f"Information about {topic} in {subject or 'general context'}",
            target_audience="General audience",
            sections=[
                Section(title="Overview", description="General information about the topic"),
                Section(title="Key Points", description="Important aspects to consider"),
                Section(title="Conclusion", description="Summary of main points")
            ],
            conclusion="This guide provides an overview of the topic."
        )
        return minimal_guide.model_dump_json(indent=2)
    
    # Create prompt for guide generation
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert content synthesizer. Create a comprehensive guide from the provided information.
        
        The guide must follow this EXACT JSON structure:
        {
            "title": "Clear and descriptive title",
            "introduction": "Comprehensive introduction to the topic",
            "target_audience": "Clear description of who this guide is for",
            "sections": [
                {
                    "title": "Section Title",
                    "description": "Detailed content for this section"
                }
            ],
            "conclusion": "Summary and key takeaways"
        }
        
        Create 3-5 well-structured sections based on the content.
        Return ONLY valid JSON that matches the GuideOutline model structure.
        Do not include any text before or after the JSON."""),
        
        ("human", """Create a comprehensive guide about: {topic}
        Query focus: {query}
        Subject area: {subject}
        
        Based on this retrieved information:
        {content}
        
        Return ONLY the JSON structure, nothing else.""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic,
        "query": query,
        "subject": subject or topic,
        "content": combined_content[:4000]  # Limit content to avoid token limits
    })
    
    # Extract content from result
    guide_json = result.content if hasattr(result, 'content') else str(result)
    
    # Clean JSON string (remove any non-JSON content)
    try:
        # Find the first { and last } to extract JSON
        start_idx = guide_json.find('{')
        end_idx = guide_json.rfind('}')
        if start_idx != -1 and end_idx != -1:
            guide_json = guide_json[start_idx:end_idx+1]
    except:
        pass
    
    # Validate and format as GuideOutline
    try:
        guide_data = json.loads(guide_json)
        
        # Ensure all required fields are present with proper types
        sections_data = guide_data.get("sections", [])
        if not sections_data:
            sections_data = [
                {"title": "Overview", "description": "General overview of the topic"},
                {"title": "Details", "description": "Detailed information"},
                {"title": "Applications", "description": "Practical applications"}
            ]
        
        # Create GuideOutline object
        guide_outline = GuideOutline(
            title=guide_data.get("title", f"Comprehensive Guide to {topic}"),
            introduction=guide_data.get("introduction", f"This guide provides detailed information about {topic}"),
            target_audience=guide_data.get("target_audience", f"Professionals and enthusiasts interested in {subject or topic}"),
            sections=[Section(**s) if isinstance(s, dict) else s for s in sections_data],
            conclusion=guide_data.get("conclusion", "This guide covered the essential aspects of the topic")
        )
        
        # Return as formatted JSON string
        return guide_outline.model_dump_json(indent=2)
        
    except Exception as e:
        print(f"Error parsing guide JSON: {e}")
        # Fallback: create a comprehensive guide structure with actual content
        fallback_guide = GuideOutline(
            title=f"Comprehensive Guide to {topic}",
            introduction=f"This guide provides information about {query} focusing on {topic} in the context of {subject or 'general knowledge'}",
            target_audience=f"Professionals and enthusiasts interested in {subject or topic}",
            sections=[
                Section(
                    title="Overview",
                    description=combined_content[:500] if combined_content else f"General overview of {topic}"
                ),
                Section(
                    title="Key Concepts",
                    description=combined_content[500:1000] if len(combined_content) > 500 else f"Important concepts related to {topic}"
                ),
                Section(
                    title="Detailed Analysis",
                    description=combined_content[1000:1500] if len(combined_content) > 1000 else f"In-depth analysis of {topic}"
                ),
                Section(
                    title="Applications",
                    description=f"Practical applications and use cases for {topic}"
                ),
                Section(
                    title="Best Practices",
                    description=f"Recommended approaches and methodologies for {topic}"
                )
            ],
            conclusion=f"This guide provided comprehensive coverage of {topic} including key concepts, applications, and best practices."
        )
        
        return fallback_guide.model_dump_json(indent=2)
