from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from typing import List, Dict, Union, Optional
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from progetto_crew_flows.models import GuideOutline, Section

load_dotenv()

"""
Ensure FAISS persistence directory is stable regardless of the current working directory.
We compute the project directory as 3 levels above this file: src/progetto_crew_flows/tools/ -> project root
Resulting path: <project_root>/RAG_database
Optionally overridden via RAG_DB_DIR environment variable.
"""
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PERSIST_DIR = os.getenv("RAG_DB_DIR") or str(BASE_DIR / "RAG_database")

@dataclass
class Settings:
    # Persistenza FAISS
    persist_dir: str = DEFAULT_PERSIST_DIR
    # Text splitting - optimized for controlled document lengths
    chunk_size: int = 200          # Smaller chunks for better precision
    chunk_overlap: int = 30        # Reduced overlap for efficiency
    # Retriever (MMR) - optimized for diverse retrieval
    search_type: str = "mmr"       # MMR for diversity (or 'similarity')
    k: int = 3                     # Increased results for better coverage
    fetch_k: int = 20              # More candidates for MMR selection
    mmr_lambda: float = 0.4        # Balanced relevance/diversity
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "LMSTUDIO_MODEL"  # nome del modello in LM Studio, via env var


SETTINGS = Settings()
API_VERSION = os.getenv("AZURE_API_VERSION")
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
# Support both CHAT_MODEL and legacy MODEL env vars for chat deployment
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL")
# Embedding deployment name must be an Azure deployment
# Initialize embeddings
EMBEDDINGS = AzureOpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=API_KEY,
    openai_api_version=API_VERSION,
    azure_endpoint=CLIENT_AZURE
)

# RAG Tools
# Tool per la generazione di documenti e store in vector database

@tool
def generate_documents(topics: List[str], docs_per_topic: int = 1, max_tokens_per_doc: int = 600, batch_size: int = 3, delay_between_batches: float = 2.0) -> str:
    """Generate multiple optimized documents for specific topics with controlled length and rate limiting
    
    Args:
        topics: List of topics to generate documents for
        docs_per_topic: Number of documents to generate per topic (default: 1)
        max_tokens_per_doc: Maximum tokens per document for efficiency (default: 600)
        batch_size: Number of topics to process in each batch (default: 3)
        delay_between_batches: Delay in seconds between batches (default: 2.0)
    
    Returns:
        JSON string containing list of document dictionaries with controlled length and diverse perspectives
    """
    
    # Safety check: limit max execution time and document count
    MAX_EXECUTION_TIME = 300  # 5 minutes max
    MAX_TOTAL_DOCUMENTS = 100  # Never generate more than 100 documents
    
    start_time = time.time()
    
    if len(topics) * docs_per_topic > MAX_TOTAL_DOCUMENTS:
        print(f"⚠️ Warning: Requested {len(topics) * docs_per_topic} documents exceeds limit of {MAX_TOTAL_DOCUMENTS}")
        print(f"   Reducing docs_per_topic to maintain limit...")
        docs_per_topic = min(docs_per_topic, MAX_TOTAL_DOCUMENTS // len(topics))
    
    print(f"🚀 Starting document generation for {len(topics)} topics ({docs_per_topic} docs each)")
    print(f"   Maximum execution time: {MAX_EXECUTION_TIME}s")
    print(f"   Maximum total documents: {MAX_TOTAL_DOCUMENTS}")

    llm = AzureChatOpenAI(
        deployment_name=CHAT_MODEL,
        openai_api_version=API_VERSION,
        azure_endpoint=CLIENT_AZURE,
        openai_api_key=API_KEY,
        temperature=0.2,  # Slightly higher for diversity between docs
        max_tokens=max_tokens_per_doc,  # Control output length
        request_timeout=30  # Add timeout for reliability
        )

    documents = []
    start_time = time.time()
    
    # Document generation strategies for efficiency and diversity
    doc_strategies = [
        {
            "focus": "fundamentals",
            "system_prompt": "You are a technical educator. Create concise, structured content focusing on core concepts and definitions.",
            "user_prompt": "Generate a focused overview of {topic} covering: 1) Key definitions 2) Core principles 3) Main applications. Keep it concise but comprehensive.",
            "doc_type": "fundamentals"
        },
        {
            "focus": "practical",
            "system_prompt": "You are a practical expert. Create actionable content with real-world examples and implementations.",
            "user_prompt": "Generate practical insights about {topic} covering: 1) Real-world use cases 2) Implementation examples 3) Best practices. Focus on actionable information.",
            "doc_type": "practical"
        },
        {
            "focus": "advanced",
            "system_prompt": "You are a research specialist. Create in-depth content covering advanced concepts and recent developments.",
            "user_prompt": "Generate advanced analysis of {topic} covering: 1) Recent developments 2) Technical challenges 3) Future trends. Focus on cutting-edge insights.",
            "doc_type": "advanced"
        },
        {
            "focus": "comparative",
            "system_prompt": "You are a comparative analyst. Create content that compares and contrasts different approaches or methodologies.",
            "user_prompt": "Generate comparative analysis of {topic} covering: 1) Different approaches 2) Advantages/disadvantages 3) When to use each. Focus on decision-making insights.",
            "doc_type": "comparative"
        }
    ]
    
    # Process topics in batches to avoid rate limits
    total_batches = (len(topics) + batch_size - 1) // batch_size
    
    for batch_num, batch_start in enumerate(range(0, len(topics), batch_size)):
        batch_topics = topics[batch_start:batch_start + batch_size]
        print(f"📦 Processing batch {batch_num + 1}/{total_batches}: {len(batch_topics)} topics")
        
        for topic in batch_topics:
            # Safety check: ensure we don't exceed max execution time
            if time.time() - start_time > MAX_EXECUTION_TIME:
                print(f"⚠️ Maximum execution time ({MAX_EXECUTION_TIME}s) reached. Stopping generation.")
                break
                
            print(f"   🔄 Generating {docs_per_topic} optimized documents for: {topic}")
            
            # Select strategies based on docs_per_topic
            selected_strategies = doc_strategies[:docs_per_topic]
            
            for doc_num, strategy in enumerate(selected_strategies):
                # Double safety check for time limit
                if time.time() - start_time > MAX_EXECUTION_TIME:
                    print(f"⚠️ Time limit reached during document generation. Stopping.")
                    break
                prompt = ChatPromptTemplate.from_messages([
                    ("system", strategy["system_prompt"]),
                    ("human", strategy["user_prompt"].format(topic=topic))
                ])
                
                try:
                    chain = prompt | llm
                    result = chain.invoke({"topic": topic})
                    
                    content = result.content if hasattr(result, 'content') else str(result)
                    
                    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(content) // 4
                    
                    documents.append({
                        "topic": topic,
                        "content": content,
                        "source": f"generated_{topic}_{strategy['focus']}_doc{doc_num + 1}",
                        "doc_type": strategy["doc_type"],
                        "focus_area": strategy["focus"],
                        "estimated_tokens": estimated_tokens,
                        "generation_strategy": f"{strategy['focus']}_optimized",
                        "batch_number": batch_num + 1
                    })
                    
                    print(f"      ✓ {strategy['focus']} document (~{estimated_tokens} tokens)")
                    
                    # Small delay between documents to avoid overwhelming the API
                    if doc_num < len(selected_strategies) - 1:
                        time.sleep(0.5)
                    
                except Exception as e:
                    print(f"      ✗ Error generating {strategy['focus']} document: {e}")
                    # Add a fallback minimal document
                    documents.append({
                        "topic": topic,
                        "content": f"Brief overview of {topic} - content generation failed but topic recorded for database.",
                        "source": f"fallback_{topic}_doc{doc_num + 1}",
                        "doc_type": "fallback",
                        "focus_area": strategy["focus"],
                        "estimated_tokens": 20,
                        "generation_strategy": "fallback",
                        "batch_number": batch_num + 1
                    })
        
        # Delay between batches to respect rate limits
        if batch_num < total_batches - 1:
            print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    total_tokens = sum(doc.get("estimated_tokens", 0) for doc in documents)
    generation_time = time.time() - start_time
    successful_docs = len([d for d in documents if d.get("doc_type") != "fallback"])
    
    print(f"\n📊 Generation Summary:")
    print(f"   ✓ {len(documents)} total documents ({successful_docs} successful, {len(documents)-successful_docs} fallback)")
    print(f"   ✓ {len(topics)} topics processed in {total_batches} batches")
    print(f"   ✓ Total estimated tokens: {total_tokens:,} (avg {total_tokens//len(documents) if documents else 0} per doc)")
    print(f"   ✓ Generation time: {generation_time:.2f}s (avg {generation_time/len(documents):.2f}s per doc)")
    print(f"   ✓ Token efficiency: {total_tokens/generation_time:.0f} tokens/second")
    
    # Return as JSON string for easy transfer between tasks
    import json
    return json.dumps(documents, ensure_ascii=False, indent=2)

@tool
def create_vectordb() -> str:
    '''Create or initialize the Vector Database to be used for RAG with WebRAGFlow.'''
    
    # Create directory if it doesn't exist
    Path(SETTINGS.persist_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize with empty documents
    initial_doc = Document(
        page_content="Vector database initialized",
        metadata={"source": "system", "topic": "initialization"}
    )

    vector_store = FAISS.from_documents([initial_doc], EMBEDDINGS)
    vector_store.save_local(SETTINGS.persist_dir)

    return f"Vector database created successfully at {SETTINGS.persist_dir}"

@tool
def store_in_vectordb(content: str, topic: Optional[str] = None) -> str:
    """Store generated content in the FAISS vector database for use with WebRAGFlow
    
    Args:
        content: Either a JSON string containing a list of documents, or plain text content
        topic: Optional topic name (used only if content is plain text)
    
    Returns:
        Success message with details about stored documents
    """
    
    documents = []
    
    # Try to parse content as JSON first (from generate_documents output)
    try:
        import json
        if isinstance(content, str) and (content.strip().startswith('[') or content.strip().startswith('{')):
            parsed_content = json.loads(content)
            
            # Handle case where it's wrapped in another dict
            if isinstance(parsed_content, dict) and 'content' in parsed_content:
                parsed_content = parsed_content['content']
            
            # Handle list of documents
            if isinstance(parsed_content, list):
                for item in parsed_content:
                    if isinstance(item, dict) and 'content' in item:
                        # Create metadata with string conversion for compatibility
                        metadata = {
                            "topic": str(item.get("topic", topic or "unknown")),
                            "source": str(item.get("source", "generated"))
                        }
                        # Add additional metadata fields if present, converting to strings
                        for key, value in item.items():
                            if key not in ["content", "topic", "source"]:
                                metadata[key] = str(value)
                                
                        doc = Document(
                            page_content=item.get("content", ""),
                            metadata=metadata
                        )
                        documents.append(doc)
            else:
                # Single document as dict
                if isinstance(parsed_content, dict) and 'content' in parsed_content:
                    metadata = {
                        "topic": str(parsed_content.get("topic", topic or "unknown")),
                        "source": str(parsed_content.get("source", "generated"))
                    }
                    doc = Document(
                        page_content=parsed_content.get("content", ""),
                        metadata=metadata
                    )
                    documents.append(doc)
    except (json.JSONDecodeError, ValueError):
        # Content is not JSON, treat as plain text
        pass
    
    # If no documents were parsed from JSON, treat as plain text
    if not documents:
        doc = Document(
            page_content=str(content),
            metadata={"topic": topic or "general", "source": f"generated_{topic or 'content'}.md"}
        )
        documents.append(doc)
    
    if not documents:
        return "Error: No valid documents found to store"
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    # Load or create vector store
    persist_dir = SETTINGS.persist_dir
    try:
        if Path(persist_dir).exists() and Path(persist_dir, "index.faiss").exists():
            vector_store = FAISS.load_local(
                persist_dir,
                EMBEDDINGS,
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            vector_store = FAISS.from_documents(chunks, EMBEDDINGS)

        # Save vector store
        vector_store.save_local(persist_dir)
        
        topics_stored = set([doc.metadata.get("topic", "unknown") for doc in documents])
        return f"Successfully stored {len(chunks)} chunks from {len(documents)} documents for topics: {', '.join(topics_stored)}"
    
    except Exception as e:
        return f"Error storing documents in vector database: {str(e)}"
    vector_store.save_local(persist_dir)
    
    topics_stored = set([doc.metadata.get("topic", "unknown") for doc in documents])
    return f"Successfully stored {len(chunks)} chunks for topics: {', '.join(topics_stored)}"

@tool
def retrieve_from_vectordb(query: str, topic: Optional[str] = None, subject: Optional[str] = None, k: Optional[int] = None) -> str:
    """
    Retrieve raw information from the vector database without formatting.
    Returns raw documents for further processing.
    """
    
    # Load FAISS index from correct path
    persist_dir = SETTINGS.persist_dir
    if not Path(persist_dir).exists() or not Path(persist_dir, "index.faiss").exists():
        return json.dumps([{"error": "Vector database not found", "content": "Please ensure RAG database is properly initialized."}])
    
    vector_store = FAISS.load_local(
        persist_dir,
        EMBEDDINGS,
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
    
    llm = AzureChatOpenAI(
        deployment_name=CHAT_MODEL,
        openai_api_version=API_VERSION,
        azure_endpoint=CLIENT_AZURE,
        openai_api_key=API_KEY,
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
