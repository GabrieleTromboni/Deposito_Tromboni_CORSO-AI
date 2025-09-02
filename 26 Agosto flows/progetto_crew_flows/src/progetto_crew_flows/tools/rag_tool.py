import json
import time
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from typing import List, Optional, Union
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from progetto_crew_flows.models import GuideOutline, Section
import uuid
from datetime import datetime

# Qdrant vector database client and models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

# Import SSL configuration
try:
    from .ssl_config import configure_azure_openai_ssl
    configure_azure_openai_ssl()
except ImportError:
    print("⚠️ SSL configuration not available")

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PERSIST_DIR = os.getenv("RAG_DB_DIR") or str(BASE_DIR / "RAG_database")

@dataclass
class Settings:
    persist_dir: str = DEFAULT_PERSIST_DIR
    chunk_size: int = 800  # Increased for richer content (was 300)
    chunk_overlap: int = 100  # Increased proportionally (was 50)
    search_type: str = "mmr"
    k: int = 3
    fetch_k: int = 10
    mmr_lambda: float = 0.4

@dataclass
# Class di setting degli iperparametri per utilizzo di Qdrant
class QdrantSetting:
    qdrant_url: str = os.getenv("QDRANT_URL")
    persist_dir: str = os.getenv("QDRANT_COLLECTION")
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_n_semantic: int = 30
    top_n_text: int = 100
    final_k: int = 6
    alpha: float = 0.75
    text_boost: float = 0.2
    use_mmr: bool = True
    mmr_lambda: float = 0.6

SETTINGS = QdrantSetting()

# Azure OpenAI configuration
API_VERSION = os.getenv("AZURE_API_VERSION")
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL")

# Initialize embeddings with error handling
try:
    EMBEDDINGS = AzureOpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=API_KEY,
        openai_api_version=API_VERSION,
        azure_endpoint=CLIENT_AZURE
    )
    print("✅ Azure OpenAI embeddings initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Azure OpenAI embeddings initialization failed: {e}")
    EMBEDDINGS = None

def safe_embeddings_check():
    """Check if embeddings are available and working"""
    if EMBEDDINGS is None:
        return False
    try:
        # Try a simple embedding to test connectivity
        test_result = EMBEDDINGS.embed_query("test")
        return len(test_result) > 0
    except Exception as e:
        print(f"⚠️ Embeddings test failed: {e}")
        return False

def parse_topic_info(topic_str: str):
    """Parse topic string to extract subject and specific topic"""
    if " - " in topic_str:
        parts = topic_str.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return "general", topic_str.strip()

def create_document_filename(subject: str, topic: str, doc_type: str, doc_index: int):
    """Create a standardized filename for document JSON files"""
    # Clean up strings for filename
    safe_subject = "".join(c for c in subject if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_doc_type = "".join(c for c in doc_type if c.isalnum() or c in (' ', '-', '_')).strip()
    
    # Replace spaces with underscores
    safe_subject = safe_subject.replace(' ', '_')
    safe_topic = safe_topic.replace(' ', '_')
    safe_doc_type = safe_doc_type.replace(' ', '_')
    
    return f"{safe_subject}_{safe_topic}_{safe_doc_type}_{doc_index}.json"

def normalize_input(value):
    """Normalize input from CrewAI agents that might wrap strings in dict format or JSON"""
    if value is None:
        return None
    
    if isinstance(value, str):
        # Try to parse as JSON if it looks like JSON
        if value.strip().startswith('{') and value.strip().endswith('}'):
            try:
                parsed = json.loads(value)
                return parsed
            except json.JSONDecodeError:
                return value
        return value
    
    if isinstance(value, dict):
        # Handle specific case for retrieved_info from RAG tool
        if 'content' in value and isinstance(value['content'], str):
            return value['content']
            
        # Handle CrewAI wrapped inputs like {'description': 'actual_value', 'type': 'str'}
        if 'description' in value and isinstance(value['description'], str):
            return value['description']
        
        # Handle simple wrapped string values like {'type': 'str', 'value': 'content'}
        if 'value' in value and isinstance(value['value'], str):
            return value['value']
        
        # Handle CrewAI tool metadata case like {'description': None, 'type': 'Union[str, dict]'}
        if 'description' in value and 'type' in value:
            if value['description'] is None:
                # This is tool metadata, not actual content - return empty or error
                print(f"⚠️ Received tool metadata instead of content: {value}")
                return ""
        
        # If it's already a proper dict structure (like from retrieve_from_vectordb), return as-is
        return value
        
        # Handle other dict structures - try to extract string content
        if len(value) == 1:
            first_value = list(value.values())[0]
            if isinstance(first_value, str):
                return first_value
        
        # If it's a complex dict, convert to JSON string
        try:
            return json.dumps(value, ensure_ascii=False)
        except:
            return str(value)
    
    return str(value)

def get_qdrant_client(settings: QdrantSetting) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

@tool
def generate_documents_as_files(topics: List[str], docs_per_topic: int = 3, max_tokens_per_doc: int = 800, batch_size: int = 2, delay_between_batches: float = 2.0) -> str:
    """Generate documents for specified topics and save each as individual JSON files"""
    
    MAX_EXECUTION_TIME = 300
    MAX_TOTAL_DOCUMENTS = 100
    
    start_time = time.time()
    
    if len(topics) * docs_per_topic > MAX_TOTAL_DOCUMENTS:
        print(f"⚠️ Warning: Requested {len(topics) * docs_per_topic} documents exceeds limit of {MAX_TOTAL_DOCUMENTS}")
        docs_per_topic = min(docs_per_topic, MAX_TOTAL_DOCUMENTS // len(topics))
    
    # Create documents directory structure
    docs_dir = Path(SETTINGS.persist_dir) / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting document generation for {len(topics)} topics ({docs_per_topic} docs each)")
    print(f"📁 Documents will be saved to: {docs_dir}")

    llm = AzureChatOpenAI(
        deployment_name=CHAT_MODEL,
        openai_api_version=API_VERSION,
        azure_endpoint=CLIENT_AZURE,
        openai_api_key=API_KEY,
        temperature=0.2,
        max_tokens=max_tokens_per_doc,
        request_timeout=30
    )

    generated_files = []
    
    # Define comprehensive document generation strategies for each topic
    doc_strategies = [
        {
            "focus": "comprehensive_overview",
            "system_prompt": "You are an expert knowledge creator. Create comprehensive, detailed content with specific facts, key concepts, and practical information.",
            "user_prompt": """Create a comprehensive guide about {topic}. Include:
1. Key definitions and concepts
2. Historical background and development
3. Current status and trends
4. Important figures and organizations
5. Key statistics and facts
6. Benefits and challenges
7. Future outlook
8. Practical applications

Make it informative and specific to {topic}. Use concrete examples and data when possible.""",
            "doc_type": "comprehensive_overview"
        },
        {
            "focus": "detailed_analysis",
            "system_prompt": "You are a subject matter expert. Create detailed analytical content focusing on specific aspects and technical details.",
            "user_prompt": """Provide a detailed analysis of {topic} covering:
1. Technical specifications and characteristics
2. Comparative analysis with similar topics
3. Strengths and weaknesses
4. Market position and influence
5. Key stakeholders and players
6. Performance metrics and indicators
7. Best practices and recommendations
8. Case studies and examples

Focus on depth and specificity for {topic}.""",
            "doc_type": "detailed_analysis"
        }
    ]
    
    total_batches = (len(topics) + batch_size - 1) // batch_size
    
    for batch_num, batch_start in enumerate(range(0, len(topics), batch_size)):
        batch_end = min(batch_start + batch_size, len(topics))
        batch_topics = topics[batch_start:batch_end]
        
        print(f"\n📦 Processing batch {batch_num + 1}/{total_batches}: {len(batch_topics)} topics")
        
        for topic_str in batch_topics:
            subject, topic = parse_topic_info(topic_str)
            print(f"   🔹 Generating {docs_per_topic} documents for: {subject} -> {topic}")
            
            # Create subject directory
            subject_dir = docs_dir / subject
            subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate documents for this topic using different strategies
            for doc_index in range(docs_per_topic):
                if time.time() - start_time > MAX_EXECUTION_TIME:
                    print(f"⏰ Time limit reached. Stopping generation.")
                    break
                
                # Select strategy (cycle through available strategies)
                strategy = doc_strategies[doc_index % len(doc_strategies)]
                
                # Create the chat prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", strategy["system_prompt"]),
                    ("user", strategy["user_prompt"])
                ])
                
                try:
                    # Generate the document
                    chain = prompt | llm
                    response = chain.invoke({"topic": topic})
                    content = response.content
                    
                    # Create document data structure
                    doc_data = {
                        "id": str(uuid.uuid4()),
                        "subject": subject,
                        "topic": topic,
                        "full_topic": topic_str,
                        "doc_type": strategy["doc_type"],
                        "doc_index": doc_index,
                        "content": content,
                        "estimated_tokens": len(content.split()) * 1.3,  # Rough estimate
                        "generated_at": datetime.now().isoformat(),
                        "strategy": strategy["focus"],
                        "metadata": {
                            "subject": subject,
                            "topic": topic,
                            "doc_type": strategy["doc_type"],
                            "source": "ai_generated"
                        }
                    }
                    
                    # Create filename and save the document
                    filename = create_document_filename(subject, topic, strategy["doc_type"], doc_index)
                    file_path = subject_dir / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(doc_data, f, ensure_ascii=False, indent=2)
                    
                    generated_files.append(str(file_path))
                    print(f"      ✅ Saved: {filename} ({len(content)} chars)")
                    
                except Exception as e:
                    print(f"      ❌ Error generating document {doc_index} for {topic}: {e}")
                    # Create fallback document
                    fallback_data = {
                        "id": str(uuid.uuid4()),
                        "subject": subject,
                        "topic": topic,
                        "full_topic": topic_str,
                        "doc_type": "fallback",
                        "doc_index": doc_index,
                        "content": f"Basic information about {topic}. This is a fallback document generated due to generation error.",
                        "estimated_tokens": 20,
                        "generated_at": datetime.now().isoformat(),
                        "strategy": "fallback",
                        "error": str(e),
                        "metadata": {
                            "subject": subject,
                            "topic": topic,
                            "doc_type": "fallback",
                            "source": "ai_generated_fallback"
                        }
                    }
                    
                    filename = create_document_filename(subject, topic, "fallback", doc_index)
                    file_path = subject_dir / filename
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(fallback_data, f, ensure_ascii=False, indent=2)
                    
                    generated_files.append(str(file_path))
        
        # Add delay between batches to respect rate limits
        if batch_num < total_batches - 1 and delay_between_batches > 0:
            print(f"   ⏱️ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    generation_time = time.time() - start_time
    
    print(f"\n📊 Generation Summary:")
    print(f"   ✓ {len(generated_files)} individual document files created")
    print(f"   ✓ Saved to: {docs_dir}")
    print(f"   ✓ Generation time: {generation_time:.2f}s")
    
    # Return summary information
    return json.dumps({
        "status": "success",
        "total_files": len(generated_files),
        "files": generated_files,
        "docs_directory": str(docs_dir),
        "generation_time": generation_time,
        "topics_processed": len(topics),
        "docs_per_topic": docs_per_topic
    }, ensure_ascii=False, indent=2)
 
@tool
def recreate_collection_for_rag(vector_size: int, qdrant_url: Optional[str] = None, collection_name: Optional[str] = None) -> str:
    """
    Create or recreate a Qdrant collection optimized for RAG (Retrieval-Augmented Generation).
    Works in sync with generate_documents_as_files output format.
    
    Returns:
        JSON string with collection creation status and details
    """
    print(f"\n🔧 RECREATE_COLLECTION_FOR_RAG:")
    
    # Use settings or provided parameters
    settings = SETTINGS
    if qdrant_url:
        settings.qdrant_url = qdrant_url
    if collection_name:
        settings.collection = collection_name
        
    print(f"   Qdrant URL: {settings.qdrant_url}")
    print(f"   Collection: {settings.persist_dir}")
    
    # Validate Qdrant connection parameters
    if not settings.qdrant_url:
        return json.dumps({
            "status": "error",
            "message": "Qdrant URL not configured. Set QDRANT_URL environment variable."
        }, ensure_ascii=False, indent=2)
    
    if not settings.persist_dir:
        return json.dumps({
            "status": "error", 
            "message": "Collection name not configured. Set QDRANT_COLLECTION environment variable."
        }, ensure_ascii=False, indent=2)
    
    try:
        # Initialize Qdrant client
        client = get_qdrant_client(settings)
        print(f"   📡 Connected to Qdrant at {settings.qdrant_url}")
        
        # Get embedding dimensions from Azure OpenAI
        if not safe_embeddings_check():
            return json.dumps({
                "status": "error",
                "message": "Azure OpenAI embeddings not available"
            }, ensure_ascii=False, indent=2)
        
        # Test embedding to get actual vector size (override provided parameter)
        test_embedding = EMBEDDINGS.embed_query("test")
        actual_vector_size = len(test_embedding)
        print(f"   📏 Detected vector dimensions: {actual_vector_size} (provided: {vector_size})")
        
        # Use actual dimensions instead of provided parameter
        vector_size = actual_vector_size
        client.recreate_collection(
            collection_name=settings.persist_dir,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
                ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=2  # parallelismo/segmentazione iniziale
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
            ),
        )
        print(f"   ✅ Collection '{settings.persist_dir}' recreated successfully")

        # Create payload indices for optimized filtering and search
        # Text index for full-text search (BM25)
        client.create_payload_index(
            collection_name=settings.persist_dir,
            field_name="content",  # Match document content field
            field_schema=PayloadSchemaType.TEXT
        )
        
        # Keyword indices for exact matching and fast filtering
        keyword_fields = ["subject", "topic", "doc_type", "source", "doc_id"]
        for field in keyword_fields:
            try:
                client.create_payload_index(
                    collection_name=settings.persist_dir,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                print(f"   ⚠️ Warning: Could not create index for field '{field}': {e}")
        
        print(f"   📊 Created {len(keyword_fields) + 1} payload indices")
        
        # Return success status
        return json.dumps({
            "status": "success",
            "collection_name": settings.persist_dir,
            "vector_size": vector_size,
            "qdrant_url": settings.qdrant_url,
            "indices_created": len(keyword_fields) + 1,
            "message": f"Qdrant collection '{settings.persist_dir}' ready for RAG operations"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"   ❌ Error creating Qdrant collection: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to create Qdrant collection: {str(e)}",
            "collection_name": settings.persist_dir if settings.persist_dir else "unknown"
        }, ensure_ascii=False, indent=2)

@tool
def store_documents_in_qdrant() -> str:
    """Load individual JSON document files and store them in the Qdrant vector database"""
    
    print(f"\n🔧 STORE_DOCUMENTS_IN_QDRANT:")
    
    # Validate Qdrant configuration
    settings = SETTINGS
    if not settings.qdrant_url or not settings.persist_dir:
        return json.dumps({
            "status": "error",
            "message": "Qdrant URL or collection name not configured. Check QDRANT_URL and QDRANT_COLLECTION environment variables."
        }, ensure_ascii=False, indent=2)
    
    # Check embeddings availability
    if not safe_embeddings_check():
        return json.dumps({
            "status": "error", 
            "message": "Azure OpenAI embeddings not available"
        }, ensure_ascii=False, indent=2)
    
    docs_dir = Path(SETTINGS.persist_dir) / "documents"
    
    if not docs_dir.exists():
        return json.dumps({
            "status": "error",
            "message": "Documents directory not found. Run generate_documents_as_files first.",
            "path": str(docs_dir)
        }, ensure_ascii=False, indent=2)
    
    # Find all JSON document files
    json_files = list(docs_dir.rglob("*.json"))
    print(f"   📄 Found {len(json_files)} document files")
    
    if not json_files:
        return json.dumps({
            "status": "error",
            "message": "No document files found",
            "path": str(docs_dir)
        }, ensure_ascii=False, indent=2)
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client(settings)
        print(f"   📡 Connected to Qdrant collection: {settings.persist_dir}")

        # Load all documents
        documents = []
        file_stats = {}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                # Create document with metadata matching generate_documents_as_files format
                doc = Document(
                    page_content=doc_data['content'],
                    metadata={
                        "doc_id": doc_data.get('id', str(uuid.uuid4())),
                        "subject": doc_data.get('subject', 'unknown'),
                        "topic": doc_data.get('topic', 'unknown'),
                        "doc_type": doc_data.get('doc_type', 'unknown'),
                        "source": doc_data.get('metadata', {}).get('source', 'ai_generated'),
                        "generated_at": doc_data.get('generated_at'),
                        "strategy": doc_data.get('strategy', 'unknown'),
                        "file_path": str(json_file)
                    }
                )
                documents.append(doc)
                
                # Track file statistics
                key = f"{doc_data.get('subject', 'unknown')}_{doc_data.get('topic', 'unknown')}"
                if key not in file_stats:
                    file_stats[key] = 0
                file_stats[key] += 1
                
                print(f"   ✅ Loaded: {json_file.stem} ({len(doc_data['content'])} chars)")
                
            except Exception as e:
                print(f"   ❌ Error loading {json_file}: {e}")
                continue
        
        if not documents:
            return json.dumps({
                "status": "error",
                "message": "No valid documents could be loaded",
                "files_found": len(json_files)
            }, ensure_ascii=False, indent=2)
        
        print(f"\n   📊 Document distribution:")
        for key, count in file_stats.items():
            print(f"      {key}: {count} documents")
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"   🔪 Split into {len(chunks)} chunks (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})")
        
        # Generate embeddings and create points for Qdrant
        points = []
        batch_size = 10  # Process in batches to avoid memory issues
        
        print(f"   🧮 Generating embeddings and creating Qdrant points...")
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            # Generate embeddings for batch
            texts = [chunk.page_content for chunk in batch_chunks]
            embeddings = EMBEDDINGS.embed_documents(texts)
            
            # Create Qdrant points
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                point_id = i + j
                
                # Convert numpy float32 to regular float for JSON serialization
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif isinstance(embedding, list):
                    embedding = [float(x) for x in embedding]
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "subject": chunk.metadata.get("subject", "unknown"),
                        "topic": chunk.metadata.get("topic", "unknown"), 
                        "doc_type": chunk.metadata.get("doc_type", "unknown"),
                        "doc_id": chunk.metadata.get("doc_id", "unknown"),
                        "source": chunk.metadata.get("source", "ai_generated"),
                        "generated_at": chunk.metadata.get("generated_at"),
                        "strategy": chunk.metadata.get("strategy", "unknown"),
                        "chunk_index": point_id,
                        "content_length": len(chunk.page_content)
                    }
                )
                points.append(point)
            
            print(f"      Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
        
        # Upload points to Qdrant in batches
        upload_batch_size = 50
        total_uploaded = 0
        
        print(f"   ⬆️ Uploading {len(points)} points to Qdrant in batches of {upload_batch_size}...")
        
        for i in range(0, len(points), upload_batch_size):
            batch_points = points[i:i + upload_batch_size]
            
            try:
                client.upsert(
                    collection_name=settings.persist_dir,
                    points=batch_points
                )
                total_uploaded += len(batch_points)
                print(f"      Uploaded batch {i//upload_batch_size + 1}/{(len(points) + upload_batch_size - 1)//upload_batch_size} ({total_uploaded}/{len(points)} points)")
                
            except Exception as e:
                print(f"      ❌ Error uploading batch {i//upload_batch_size + 1}: {e}")
                continue
        
        # Create detailed summary
        chunk_stats = {}
        for chunk in chunks:
            key = f"{chunk.metadata.get('subject', 'unknown')}_{chunk.metadata.get('topic', 'unknown')}"
            if key not in chunk_stats:
                chunk_stats[key] = 0
            chunk_stats[key] += 1
        
        print(f"\n   📈 Chunk distribution:")
        for key, count in chunk_stats.items():
            print(f"      {key}: {count} chunks")
        
        print(f"\n   ✅ Successfully stored {total_uploaded} chunks in Qdrant collection '{settings.persist_dir}'")
        
        return json.dumps({
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "points_uploaded": total_uploaded,
            "files_processed": len(json_files),
            "chunk_distribution": chunk_stats,
            "document_distribution": file_stats,
            "collection_name": settings.collection,
            "qdrant_url": settings.qdrant_url
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"   ❌ Error storing documents in Qdrant: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Failed to store documents in Qdrant: {str(e)}",
            "collection_name": settings.collection if settings.collection else "unknown"
        }, ensure_ascii=False, indent=2)

@tool
def create_vectordb() -> str:
    """Create or initialize the Vector Database"""
    
    Path(SETTINGS.persist_dir).mkdir(parents=True, exist_ok=True)
    
    initial_doc = Document(
        page_content="Vector database initialized",
        metadata={"source": "system", "topic": "initialization"}
    )

    vector_store = FAISS.from_documents([initial_doc], EMBEDDINGS)
    vector_store.save_local(SETTINGS.persist_dir)

    return f"Vector database created successfully at {SETTINGS.persist_dir}"

@tool
def store_individual_documents() -> str:
    """Load individual JSON document files and store them in the FAISS vector database"""
    
    print(f"\n🔧 STORE_INDIVIDUAL_DOCUMENTS:")
    
    docs_dir = Path(SETTINGS.persist_dir) / "documents"
    
    if not docs_dir.exists():
        return json.dumps({
            "error": "Documents directory not found",
            "path": str(docs_dir),
            "message": "Run generate_documents_as_files first"
        })
    
    # Find all JSON document files
    json_files = list(docs_dir.rglob("*.json"))
    print(f"   📄 Found {len(json_files)} document files")
    
    if not json_files:
        return json.dumps({
            "error": "No document files found",
            "path": str(docs_dir)
        })
    
    # Load all documents
    documents = []
    file_stats = {}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Create Langchain Document
            document = Document(
                page_content=doc_data["content"],
                metadata={
                    "id": doc_data["id"],
                    "subject": doc_data["subject"],
                    "topic": doc_data["topic"],
                    "doc_type": doc_data["doc_type"],
                    "doc_index": doc_data["doc_index"],
                    "source": doc_data.get("strategy", "unknown"),
                    "file_path": str(json_file),
                    "generated_at": doc_data.get("generated_at", "unknown")
                }
            )
            
            documents.append(document)
            
            # Track stats
            subject = doc_data["subject"]
            topic = doc_data["topic"]
            key = f"{subject}/{topic}"
            if key not in file_stats:
                file_stats[key] = 0
            file_stats[key] += 1
            
            print(f"   ✅ Loaded: {json_file.stem} ({len(doc_data['content'])} chars)")
            
        except Exception as e:
            print(f"   ❌ Error loading {json_file}: {e}")
            continue
    
    if not documents:
        return json.dumps({
            "error": "No valid documents could be loaded",
            "files_found": len(json_files)
        })
    
    print(f"\n   📊 Document distribution:")
    for key, count in file_stats.items():
        print(f"      {key}: {count} documents")
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for structured content
    )
    chunks = splitter.split_documents(documents)
    print(f"   🔪 Split into {len(chunks)} chunks (chunk_size={SETTINGS.chunk_size}, overlap={SETTINGS.chunk_overlap})")
    
    # Debug: Show sample chunks to verify quality
    if chunks:
        print(f"   🔍 Sample chunk preview:")
        sample_chunk = chunks[0]
        content_preview = sample_chunk.page_content[:200] + "..." if len(sample_chunk.page_content) > 200 else sample_chunk.page_content
        print(f"      First chunk ({len(sample_chunk.page_content)} chars): {content_preview}")
        if len(chunks) > 1:
            sample_chunk = chunks[len(chunks)//2]
            content_preview = sample_chunk.page_content[:200] + "..." if len(sample_chunk.page_content) > 200 else sample_chunk.page_content
            print(f"      Middle chunk ({len(sample_chunk.page_content)} chars): {content_preview}")
    
    # Load or create vector store
    persist_dir = SETTINGS.persist_dir
    try:
        if Path(persist_dir, "index.faiss").exists():
            print(f"   📂 Loading existing FAISS index from {persist_dir}")
            vector_store = FAISS.load_local(persist_dir, EMBEDDINGS, allow_dangerous_deserialization=True)
            # Add new chunks to existing store
            vector_store.add_documents(chunks)
        else:
            print(f"   🆕 Creating new FAISS index")
            vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
        
        # Save the updated vector store
        vector_store.save_local(persist_dir)
        print(f"   💾 Vector store saved to {persist_dir}")
        
    except Exception as e:
        print(f"   ❌ Vector store error: {e}")
        return json.dumps({
            "error": f"Vector store error: {e}",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks)
        })
    
    # Create detailed summary
    chunk_stats = {}
    for chunk in chunks:
        subject = chunk.metadata.get("subject", "unknown")
        topic = chunk.metadata.get("topic", "unknown")
        key = f"{subject}/{topic}"
        if key not in chunk_stats:
            chunk_stats[key] = 0
        chunk_stats[key] += 1
    
    print(f"\n   📈 Chunk distribution:")
    for key, count in chunk_stats.items():
        print(f"      {key}: {count} chunks")
    
    return json.dumps({
        "status": "success",
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": len(chunks),
        "files_processed": len(json_files),
        "chunk_distribution": chunk_stats,
        "document_distribution": file_stats,
        "persist_dir": persist_dir
    }, ensure_ascii=False, indent=2)

# Keep existing retrieve function
#FAISS Retrieve

@tool
def retrieve_from_vectordb(query: str, topic: Optional[str] = None, subject: Optional[str] = None, k: int = 5) -> str:
    """Retrieve information from the vector database and return structured content"""
    
    print(f"\n🔍 RETRIEVE_FROM_VECTORDB:")
    
    # Normalize inputs to handle CrewAI wrapped parameters
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    k = normalize_input(k)
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   K: {k}")
    
    # Validate inputs
    if not query or query.strip() == "":
        return "ERROR: Query cannot be empty"
    
    # Check embeddings availability
    if not safe_embeddings_check():
        return "ERROR: Embeddings not available"
    
    # Load FAISS index from correct path
    persist_dir = SETTINGS.persist_dir
    print(f"   Persist dir: {persist_dir}")
    print(f"   Dir exists: {Path(persist_dir).exists()}")
    print(f"   Index exists: {Path(persist_dir, 'index.faiss').exists()}")
    
    if not Path(persist_dir).exists() or not Path(persist_dir, "index.faiss").exists():
        return f"ERROR: Vector database not found at {persist_dir}"
    
    try:
        vector_store = FAISS.load_local(
            persist_dir, 
            EMBEDDINGS, 
            allow_dangerous_deserialization=True
        )
        print(f"   📂 Vector store loaded successfully")
        
    except Exception as e:
        return f"ERROR: Failed to load vector store: {e}"
    
    # Build search query with both subject and topic if available
    if subject and topic:
        search_query = f"{subject} {topic} {query}"
    elif topic:
        search_query = f"{topic} {query}"
    else:
        search_query = query
    
    print(f"   🔎 Search query: '{search_query}'")
    
    # Use custom k if provided (default to 5 for focused results)
    try:
        search_k = int(k) if k else 5
    except (ValueError, TypeError):
        search_k = 5
    print(f"   📝 Search parameters: k={search_k}")
    
    try:
        docs = vector_store.similarity_search_with_score(
            search_query, 
            k=search_k * 2  # Get more docs for filtering
        )
        print(f"   🔍 Raw similarity search returned: {len(docs)} documents")
                            
    except Exception as e:
        return f"ERROR: Search failed: {e}"

    # Remove duplicates while preserving order
    seen_content = set()
    unique_docs = []
    for doc, score in docs:
        content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append((doc, score))

    docs = unique_docs[:search_k]  # Limit to requested number
    print(f"   🔹 Final unique documents: {len(docs)}")

    if not docs:
        return f"ERROR: No relevant documents found for query: {search_query}"

    # Format results as JSON structure for format_content_as_guide
    documents_list = []
    
    for i, (doc, score) in enumerate(docs, 1):
        # JSON format document
        doc_dict = {
            "content": doc.page_content,
            "score": float(score)  # Convert numpy float32 to standard float
        }
        
        # Add metadata if available
        metadata = doc.metadata
        if metadata:
            if 'subject' in metadata:
                doc_dict['subject'] = metadata['subject']
            if 'topic' in metadata:
                doc_dict['topic'] = metadata['topic']
            if 'source' in metadata:
                doc_dict['source'] = metadata['source']
        
        documents_list.append(doc_dict)
    
    print(f"   ✅ Returning {len(docs)} results as structured data")
    
    # Return JSON structure that format_content_as_guide expects
    return json.dumps({
        "documents": documents_list,
        "query": query,
        "topic": topic,
        "subject": subject,
        "total_documents": len(docs),
        "search_query": search_query
    }, ensure_ascii=False, indent=2)

# =========================
# QDRANT SEARCH TOOLS - Enhanced and Unified
# =========================

# Internal implementation functions (not decorated)
def _qdrant_semantic_search_impl(query: str, limit: int = 30, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Internal implementation for semantic search on Qdrant database"""
    
    print(f"\n🔍 QDRANT_SEMANTIC_SEARCH:")
    
    # Normalize inputs
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   Limit: {limit}")
    
    # Validate inputs
    if not query or query.strip() == "":
        return "ERROR: Query cannot be empty"
    
    # Check configuration
    settings = SETTINGS
    if not settings.qdrant_url or not settings.persist_dir:
        return "ERROR: Qdrant not configured. Check QDRANT_URL and QDRANT_COLLECTION environment variables."
    
    # Check embeddings availability
    if not safe_embeddings_check():
        return "ERROR: Azure OpenAI embeddings not available"
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client(settings)
        print(f"   📡 Connected to Qdrant collection: {settings.persist_dir}")
        
        # Build search query with context
        if subject and topic:
            search_query = f"{subject} {topic} {query}"
        elif topic:
            search_query = f"{topic} {query}"
        else:
            search_query = query
        
        print(f"   🔎 Search query: '{search_query}'")
        
        # Generate query embedding
        qv = EMBEDDINGS.embed_query(search_query)
        
        # Perform semantic search
        res = client.query_points(
            collection_name=settings.persist_dir,
            query=qv,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
                exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
            ),
        )
        
        print(f"   📊 Found {len(res.points)} semantic results")
        
        if not res.points:
            return json.dumps({
                "status": "no_results",
                "message": f"No semantic results found for query: {search_query}",
                "query": query,
                "search_query": search_query
            }, ensure_ascii=False, indent=2)
        
        # Format results for RAG crew
        documents_list = []
        for i, point in enumerate(res.points):
            doc_dict = {
                "content": point.payload.get("content", ""),
                "score": float(point.score),
                "search_type": "semantic",
                "point_id": point.id
            }
            
            # Add metadata if available
            if "subject" in point.payload:
                doc_dict["subject"] = point.payload["subject"]
            if "topic" in point.payload:
                doc_dict["topic"] = point.payload["topic"]
            if "doc_type" in point.payload:
                doc_dict["doc_type"] = point.payload["doc_type"]
            if "source" in point.payload:
                doc_dict["source"] = point.payload["source"]
            
            documents_list.append(doc_dict)
        
        print(f"   ✅ Returning {len(documents_list)} semantic results")
        
        return json.dumps({
            "status": "success",
            "search_type": "semantic",
            "documents": documents_list,
            "query": query,
            "topic": topic,
            "subject": subject,
            "total_documents": len(documents_list),
            "search_query": search_query
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"   ❌ Semantic search error: {e}")
        return json.dumps({
            "status": "error",
            "search_type": "semantic",
            "message": f"Semantic search failed: {str(e)}",
            "query": query
        }, ensure_ascii=False, indent=2)

def _qdrant_text_search_impl(query: str, limit: int = 100, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Internal implementation for text-based search on Qdrant database"""
    
    print(f"\n🔍 QDRANT_TEXT_SEARCH:")
    
    # Normalize inputs
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   Limit: {limit}")
    
    # Validate inputs
    if not query or query.strip() == "":
        return "ERROR: Query cannot be empty"
    
    # Check configuration
    settings = SETTINGS
    if not settings.qdrant_url or not settings.persist_dir:
        return "ERROR: Qdrant not configured. Check QDRANT_URL and QDRANT_COLLECTION environment variables."
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client(settings)
        print(f"   📡 Connected to Qdrant collection: {settings.persist_dir}")
        
        # Build search query with context
        if subject and topic:
            search_query = f"{subject} {topic} {query}"
        elif topic:
            search_query = f"{topic} {query}"
        else:
            search_query = query
        
        print(f"   🔎 Text search query: '{search_query}'")
        
        # Perform text-based search using scroll with MatchText filter
        matched_points = []
        next_page = None
        
        while len(matched_points) < limit:
            points, next_page = client.scroll(
                collection_name=settings.persist_dir,
                scroll_filter=Filter(
                    must=[FieldCondition(key="content", match=MatchText(text=search_query))]
                ),
                limit=min(100, limit - len(matched_points)),
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            
            matched_points.extend(points)
            
            if not next_page or len(matched_points) >= limit:
                break
        
        print(f"   📊 Found {len(matched_points)} text-based results")
        
        if not matched_points:
            return json.dumps({
                "status": "no_results",
                "message": f"No text results found for query: {search_query}",
                "query": query,
                "search_query": search_query
            }, ensure_ascii=False, indent=2)
        
        # Format results for RAG crew (simulate scores for text matches)
        documents_list = []
        for i, point in enumerate(matched_points[:limit]):
            # Create a pseudo-score based on text relevance (simple heuristic)
            content = point.payload.get("content", "")
            query_words = search_query.lower().split()
            content_words = content.lower().split()
            matches = sum(1 for word in query_words if word in content_words)
            pseudo_score = min(0.95, matches / max(len(query_words), 1))
            
            doc_dict = {
                "content": content,
                "score": pseudo_score,
                "search_type": "text",
                "point_id": point.id
            }
            
            # Add metadata if available
            if "subject" in point.payload:
                doc_dict["subject"] = point.payload["subject"]
            if "topic" in point.payload:
                doc_dict["topic"] = point.payload["topic"]
            if "doc_type" in point.payload:
                doc_dict["doc_type"] = point.payload["doc_type"]
            if "source" in point.payload:
                doc_dict["source"] = point.payload["source"]
            
            documents_list.append(doc_dict)
        
        # Sort by pseudo-score descending
        documents_list.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"   ✅ Returning {len(documents_list)} text-based results")
        
        return json.dumps({
            "status": "success",
            "search_type": "text",
            "documents": documents_list,
            "query": query,
            "topic": topic,
            "subject": subject,
            "total_documents": len(documents_list),
            "search_query": search_query
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"   ❌ Text search error: {e}")
        return json.dumps({
            "status": "error",
            "search_type": "text",
            "message": f"Text search failed: {str(e)}",
            "query": query
        }, ensure_ascii=False, indent=2)

def _qdrant_hybrid_search_impl(query: str, k: int = 6, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Internal implementation for intelligent hybrid search"""
    
    print(f"\n🔍 QDRANT_HYBRID_SEARCH:")
    
    # Normalize inputs
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   K: {k}")
    
    # Validate inputs
    if not query or query.strip() == "":
        return "ERROR: Query cannot be empty"
    
    # Check configuration
    settings = SETTINGS
    if not settings.qdrant_url or not settings.persist_dir:
        return "ERROR: Qdrant not configured. Check QDRANT_URL and QDRANT_COLLECTION environment variables."
    
    # Check embeddings availability
    if not safe_embeddings_check():
        return "ERROR: Azure OpenAI embeddings not available"
    
    try:
        # Connect to Qdrant
        client = get_qdrant_client(settings)
        print(f"   📡 Connected to Qdrant collection: {settings.persist_dir}")
        
        # Build search query with context
        if subject and topic:
            search_query = f"{subject} {topic} {query}"
        elif topic:
            search_query = f"{topic} {query}"
        else:
            search_query = query
        
        print(f"   🔎 Hybrid search query: '{search_query}'")
        
        # Phase 1: Semantic Search
        qv = EMBEDDINGS.embed_query(search_query)
        semantic_results = client.query_points(
            collection_name=settings.persist_dir,
            query=qv,
            limit=settings.top_n_semantic,
            with_payload=True,
            with_vectors=True,
            search_params=SearchParams(
                hnsw_ef=256,
                exact=False
            ),
        )
        
        print(f"   📊 Semantic phase: {len(semantic_results.points)} results")
        
        if not semantic_results.points:
            return json.dumps({
                "status": "no_results",
                "message": f"No semantic results found for query: {search_query}",
                "query": query,
                "search_query": search_query,
                "search_type": "hybrid"
            }, ensure_ascii=False, indent=2)
        
        # Phase 2: Text-based pre-filtering
        text_ids = set()
        next_page = None
        while len(text_ids) < settings.top_n_text:
            points, next_page = client.scroll(
                collection_name=settings.persist_dir,
                scroll_filter=Filter(
                    must=[FieldCondition(key="content", match=MatchText(text=search_query))]
                ),
                limit=min(100, settings.top_n_text - len(text_ids)),
                offset=next_page,
                with_payload=False,
                with_vectors=False,
            )
            text_ids.update([p.id for p in points])
            if not next_page or len(text_ids) >= settings.top_n_text:
                break
        
        print(f"   📊 Text phase: {len(text_ids)} matching IDs")
        
        # Phase 3: Score Fusion and Ranking
        # Normalize semantic scores
        semantic_scores = [p.score for p in semantic_results.points]
        if semantic_scores:
            smin, smax = min(semantic_scores), max(semantic_scores)
            def normalize_score(x):
                return 1.0 if smax == smin else (x - smin) / (smax - smin)
        else:
            def normalize_score(x):
                return 0.0
        
        # Combine scores using hybrid fusion
        hybrid_results = []
        for point in semantic_results.points:
            # Normalized semantic score
            norm_semantic = normalize_score(point.score)
            
            # Text relevance boost
            text_match = 1.0 if point.id in text_ids else 0.0
            
            # Hybrid score calculation: alpha * semantic + text_boost * text_match
            hybrid_score = (settings.alpha * norm_semantic + 
                          settings.text_boost * text_match)
            
            hybrid_results.append((point, hybrid_score, text_match > 0))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   📊 Hybrid fusion completed, top score: {hybrid_results[0][1]:.3f}")
        
        # Phase 4: MMR Diversification (if enabled)
        if settings.use_mmr and len(hybrid_results) > k:
            # Extract vectors and scores for MMR
            candidates_vectors = [result[0].vector for result in hybrid_results]
            
            # Apply MMR selection
            selected_indices = mmr_select(
                query_vec=qv,
                candidates_vecs=candidates_vectors,
                k=k,
                lambda_mult=settings.mmr_lambda
            )
            
            final_results = [hybrid_results[i] for i in selected_indices]
            print(f"   🎯 MMR diversification: selected {len(final_results)} diverse results")
        else:
            final_results = hybrid_results[:k]
            print(f"   📝 Top-K selection: selected {len(final_results)} results")
        
        # Phase 5: Format results for RAG crew
        documents_list = []
        for point, hybrid_score, has_text_match in final_results:
            doc_dict = {
                "content": point.payload.get("content", ""),
                "score": float(hybrid_score),
                "semantic_score": float(normalize_score(point.score)),
                "text_match": has_text_match,
                "search_type": "hybrid",
                "point_id": point.id
            }
            
            # Add metadata if available
            for key in ["subject", "topic", "doc_type", "source"]:
                if key in point.payload:
                    doc_dict[key] = point.payload[key]
            
            documents_list.append(doc_dict)
        
        print(f"   ✅ Returning {len(documents_list)} hybrid results")
        
        return json.dumps({
            "status": "success",
            "search_type": "hybrid",
            "documents": documents_list,
            "query": query,
            "topic": topic,
            "subject": subject,
            "total_documents": len(documents_list),
            "search_query": search_query,
            "semantic_candidates": len(semantic_results.points),
            "text_matches": len(text_ids),
            "mmr_enabled": settings.use_mmr,
            "fusion_settings": {
                "alpha": settings.alpha,
                "text_boost": settings.text_boost,
                "mmr_lambda": settings.mmr_lambda if settings.use_mmr else None
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"   ❌ Hybrid search error: {e}")
        return json.dumps({
            "status": "error",
            "search_type": "hybrid",
            "message": f"Hybrid search failed: {str(e)}",
            "query": query
        }, ensure_ascii=False, indent=2)

# Tool decorators (use internal implementations)
@tool
def qdrant_semantic_search(query: str, limit: int = 30, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Perform semantic search on Qdrant database using vector similarity"""
    return _qdrant_semantic_search_impl(query, limit, topic, subject)

@tool
def qdrant_text_search(query: str, limit: int = 100, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Perform text-based search on Qdrant database using full-text matching"""
    return _qdrant_text_search_impl(query, limit, topic, subject)

@tool
def qdrant_hybrid_search(query: str, k: int = 6, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """Perform intelligent hybrid search combining semantic and text-based approaches"""
    return _qdrant_hybrid_search_impl(query, k, topic, subject)

def mmr_select(query_vec: List[float], candidates_vecs: List[List[float]], k: int, lambda_mult: float) -> List[int]:
    """
    Select diverse results using Maximal Marginal Relevance (MMR) algorithm.
    
    MMR balances relevance to the query with diversity among selected results,
    reducing redundancy and improving information coverage.
    """
    import numpy as np
    
    if not candidates_vecs or k <= 0:
        return []
    
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)

    def cos_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    # Calculate relevance scores for all candidates
    relevance_scores = [cos_similarity(v, q) for v in V]
    
    selected = []
    remaining = set(range(len(V)))

    # Select first item with highest relevance
    if remaining:
        best_idx = max(remaining, key=lambda i: relevance_scores[i])
        selected.append(best_idx)
        remaining.remove(best_idx)

    # Iteratively select items balancing relevance and diversity
    while len(selected) < min(k, len(V)) and remaining:
        best_idx = None
        best_score = -1e9
        
        for i in remaining:
            # Calculate maximum similarity to already selected items
            max_similarity = max([cos_similarity(V[i], V[j]) for j in selected]) if selected else 0.0
            
            # MMR score: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_mult * relevance_scores[i] - (1 - lambda_mult) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    return selected

@tool
def intelligent_rag_search(query: str, k: int = 6, topic: Optional[str] = None, subject: Optional[str] = None, search_strategy: Optional[str] = "auto") -> str:
    """
    Intelligent RAG search that automatically selects the best search strategy based on query characteristics.
    
    This tool analyzes the user's query to determine the optimal search approach:
    - Hybrid search for complex conceptual queries
    - Semantic search for conceptual/relationship queries  
    - Text search for specific facts/keywords
    
    Args:
        query: User's search query
        k: Number of results to return (default: 6)
        topic: Optional topic filter
        subject: Optional subject filter
        search_strategy: Optional manual strategy override ("auto", "hybrid", "semantic", "text")
    
    Returns:
        JSON string with search results and metadata
    """
    
    print(f"\n🧠 INTELLIGENT_RAG_SEARCH:")
    
    # Normalize inputs
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    search_strategy = normalize_input(search_strategy) or "auto"
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   K: {k}")
    print(f"   Strategy: {search_strategy}")
    
    # Validate inputs
    if not query or query.strip() == "":
        return "ERROR: Query cannot be empty"
    
    # Strategy selection logic
    if search_strategy == "auto":
        selected_strategy = _analyze_query_for_strategy(query, topic, subject)
        print(f"   🎯 Auto-selected strategy: {selected_strategy}")
    else:
        selected_strategy = search_strategy.lower()
        print(f"   🎯 Manual strategy: {selected_strategy}")
    
    # Validate strategy
    if selected_strategy not in ["hybrid", "semantic", "text"]:
        selected_strategy = "hybrid"  # Default fallback
        print(f"   ⚠️ Invalid strategy, using default: {selected_strategy}")
    
    # Execute selected strategy
    try:
        if selected_strategy == "hybrid":
            result = _qdrant_hybrid_search_impl(query=query, k=k, topic=topic, subject=subject)
        elif selected_strategy == "semantic":
            result = _qdrant_semantic_search_impl(query=query, limit=k*2, topic=topic, subject=subject)
        elif selected_strategy == "text":
            result = _qdrant_text_search_impl(query=query, limit=k*2, topic=topic, subject=subject)
        else:
            # Fallback to hybrid
            result = _qdrant_hybrid_search_impl(query=query, k=k, topic=topic, subject=subject)
        
        # Parse and enhance result with strategy info
        try:
            result_data = json.loads(result)
            if isinstance(result_data, dict):
                result_data["selected_strategy"] = selected_strategy
                result_data["strategy_reasoning"] = _get_strategy_reasoning(selected_strategy, query)
                
                # Limit results to requested k for non-hybrid searches
                if selected_strategy != "hybrid" and "documents" in result_data:
                    result_data["documents"] = result_data["documents"][:k]
                    result_data["total_documents"] = len(result_data["documents"])
                
                return json.dumps(result_data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
        
        return result
        
    except Exception as e:
        print(f"   ❌ Intelligent search error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Intelligent search failed: {str(e)}",
            "query": query,
            "selected_strategy": selected_strategy
        }, ensure_ascii=False, indent=2)

def _analyze_query_for_strategy(query: str, topic: Optional[str] = None, subject: Optional[str] = None) -> str:
    """
    Analyze query characteristics to determine optimal search strategy.
    
    Returns:
        str: One of "hybrid", "semantic", "text"
    """
    query_lower = query.lower()
    
    # Text search indicators (specific facts, names, exact terms)
    text_indicators = [
        # Question words for specific facts
        "what is", "who is", "when did", "where is", "how much", "how many",
        # Specific data requests
        "definition", "define", "list", "name", "show me", "find",
        # Exact matching needs
        "exactly", "specifically", "precise", "exact",
        # Technical terms that might need exact matching
        "API", "function", "method", "class", "variable"
    ]
    
    # Semantic search indicators (concepts, relationships, understanding)
    semantic_indicators = [
        # Conceptual understanding
        "explain", "understand", "concept", "principle", "theory",
        "relationship", "compare", "difference", "similarity",
        # Abstract reasoning
        "why", "how does", "what makes", "impact", "effect", "influence",
        "analyze", "evaluate", "assess", "consider",
        # Learning-oriented
        "learn", "teach", "understand", "grasp", "comprehend"
    ]
    
    # Hybrid search indicators (complex queries needing both approaches)
    hybrid_indicators = [
        # Complex analytical requests
        "comprehensive", "detailed", "thorough", "complete", "overview",
        "summary", "guide", "tutorial", "best practices",
        # Multiple aspects
        "aspects", "factors", "considerations", "approaches", "methods",
        # Comparative analysis
        "pros and cons", "advantages", "disadvantages", "trade-offs"
    ]
    
    # Count indicators
    text_score = sum(1 for indicator in text_indicators if indicator in query_lower)
    semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
    hybrid_score = sum(1 for indicator in hybrid_indicators if indicator in query_lower)
    
    # Query length analysis (longer queries often benefit from hybrid)
    word_count = len(query.split())
    if word_count > 8:
        hybrid_score += 1
    elif word_count < 4:
        text_score += 1
    
    # Subject/topic context analysis
    if subject or topic:
        # If we have specific context, hybrid often works better
        hybrid_score += 1
    
    # Special case: questions with "how" often need semantic understanding
    if query_lower.startswith("how ") and "how much" not in query_lower and "how many" not in query_lower:
        semantic_score += 2
    
    # Select strategy based on highest score
    scores = {
        "text": text_score,
        "semantic": semantic_score,
        "hybrid": hybrid_score
    }
    
    selected = max(scores.items(), key=lambda x: x[1])
    
    # If scores are tied or very low, default to hybrid
    if selected[1] == 0 or list(scores.values()).count(selected[1]) > 1:
        return "hybrid"
    
    return selected[0]

def _get_strategy_reasoning(strategy: str, query: str) -> str:
    """
    Provide human-readable explanation for strategy selection.
    
    Args:
        strategy: Selected strategy ("hybrid", "semantic", "text")
        query: Original query
        
    Returns:
        str: Explanation of why this strategy was chosen
    """
    if strategy == "hybrid":
        return (
            "Hybrid search selected for comprehensive results combining semantic understanding "
            "with keyword matching. Best for complex queries requiring both conceptual depth "
            "and factual precision."
        )
    elif strategy == "semantic":
        return (
            "Semantic search selected for conceptual understanding and relationship discovery. "
            "Optimal for queries requiring interpretation of meaning and context rather than "
            "exact keyword matching."
        )
    elif strategy == "text":
        return (
            "Text-based search selected for precise keyword and factual information retrieval. "
            "Best for queries seeking specific terms, names, definitions, or exact data points."
        )
    else:
        return "Strategy selection based on query analysis and optimization heuristics."

@tool
def format_content_as_guide(retrieved_info: Union[str, dict], query: str, topic: str, subject: Optional[str] = None) -> str:
    """Format retrieved content into a structured guide using clear input from retrieve_from_vectordb"""
    
    print(f"\n📝 FORMAT_CONTENT_AS_GUIDE:")
    
    # Normalize inputs to handle CrewAI wrapped parameters
    retrieved_info = normalize_input(retrieved_info)
    query = normalize_input(query)
    topic = normalize_input(topic)
    subject = normalize_input(subject)
    
    print(f"   Query: {query}")
    print(f"   Topic: {topic}")
    print(f"   Subject: {subject}")
    print(f"   Retrieved info length: {len(str(retrieved_info))}")
    
    # Validate inputs
    if not query or query.strip() == "":
        query = "Information request"
    if not topic or topic.strip() == "":
        topic = "General topic"
    
    # Check if retrieved_info contains an error
    if not retrieved_info or "ERROR:" in str(retrieved_info):
        error_msg = retrieved_info if retrieved_info else "No information retrieved"
        print(f"   ⚠️ Error in retrieved data: {error_msg}")
        
        error_guide = GuideOutline(
            title=f"Information Request: {topic}",
            introduction=f"We attempted to find information about {topic} in our database.",
            target_audience="General audience",
            sections=[
                Section(
                    title="Search Status",
                    description=f"Database search completed but encountered an issue: {error_msg}"
                ),
                Section(
                    title="Alternative Approach", 
                    description="For comprehensive information about this topic, consider using web search or consulting external sources."
                )
            ],
            conclusion="The requested information was not available in our current knowledge base."
        )
        return error_guide.model_dump_json(indent=2)
    
    # Parse the response from retrieve_from_vectordb
    print(f"   📄 Processing retrieved information...")
    
    documents = []
    
    # First, try to handle as JSON dict structure (new format)
    if isinstance(retrieved_info, dict):
        print(f"   📂 Processing dict input with keys: {list(retrieved_info.keys())}")
        
        if 'documents' in retrieved_info:
            documents = retrieved_info['documents']
            print(f"   📊 Found {len(documents)} documents in dict structure")
        else:
            # If it's a dict but not the expected structure, convert to single document
            documents = [retrieved_info]
            print(f"   📄 Converting single dict to document")
    
    # If no documents found yet, try parsing as structured text (legacy format)
    elif isinstance(retrieved_info, str):
        print(f"   📄 Processing structured text input...")
        
        current_doc = {}
        content_buffer = []
        
        lines = retrieved_info.split('\n')
        in_document = False
        
        for line in lines:
            if line.startswith('--- DOCUMENT '):
                # Save previous document if exists
                if current_doc and content_buffer:
                    current_doc['content'] = '\n'.join(content_buffer).strip()
                    documents.append(current_doc)
                
                # Start new document
                current_doc = {}
                content_buffer = []
                in_document = True
                
                # Extract score if available
                if 'Score:' in line:
                    try:
                        score_str = line.split('Score: ')[1].split(')')[0]
                        current_doc['score'] = float(score_str)
                    except:
                        current_doc['score'] = 0.0
                
            elif line.startswith('Content: '):
                content_buffer.append(line[9:])  # Remove 'Content: ' prefix
            elif line.startswith('Subject: '):
                current_doc['subject'] = line[9:]
            elif line.startswith('Topic: '):
                current_doc['topic'] = line[7:]
            elif line.startswith('Source: '):
                current_doc['source'] = line[8:]
            elif in_document and line.strip() and not line.startswith('---'):
                content_buffer.append(line)
        
        # Save last document
        if current_doc and content_buffer:
            current_doc['content'] = '\n'.join(content_buffer).strip()
            documents.append(current_doc)
        
        print(f"   📊 Extracted {len(documents)} documents from structured text")
    
    print(f"   📊 Final document count: {len(documents)}")
    
    # Combine all content
    combined_content = ""
    sources_used = []
    
    for doc in documents:
        if doc.get('content'):
            combined_content += doc['content'] + "\n\n"
            if doc.get('source'):
                sources_used.append(doc['source'])
    
    combined_content = combined_content.strip()
    print(f"   📄 Combined content length: {len(combined_content)}")
    
    if not combined_content or len(combined_content.strip()) < 50:
        print(f"   ⚠️ Insufficient content found, creating minimal guide")
        minimal_guide = GuideOutline(
            title=f"Information Overview: {topic}",
            introduction=f"This guide provides available information about {topic} in the context of {subject or 'general knowledge'}.",
            target_audience="General audience",
            sections=[
                Section(
                    title="Topic Overview",
                    description=f"We searched for information about {topic} but found limited content in our current database."
                ),
                Section(
                    title="Search Context",
                    description=f"Your query was: '{query}'. This topic may require additional research from external sources."
                ),
                Section(
                    title="Next Steps", 
                    description="Consider refining your search terms or exploring related topics that may be available in our knowledge base."
                )
            ],
            conclusion="This guide represents the available information for your query. For more comprehensive details, additional sources may be needed."
        )
        return minimal_guide.model_dump_json(indent=2)
    
    print(f"   � Creating comprehensive guide from {len(documents)} documents")
    
    # Create structured sections from the documents
    try:
        sections = []
        
        if len(documents) == 1:
            # Single document - create logical sections
            content = documents[0]['content']
            sections.append(Section(
                title="Overview",
                description=content[:800] if len(content) > 800 else content
            ))
            if len(content) > 800:
                sections.append(Section(
                    title="Additional Details",
                    description=content[800:1600] if len(content) > 1600 else content[800:]
                ))
        
        elif len(documents) == 2:
            sections.append(Section(
                title="Primary Information",
                description=documents[0]['content'][:800]
            ))
            sections.append(Section(
                title="Supporting Information", 
                description=documents[1]['content'][:800]
            ))
        
        else:  # 3 or more documents
            sections.append(Section(
                title="Overview",
                description=documents[0]['content'][:600]
            ))
            sections.append(Section(
                title="Key Information",
                description=documents[1]['content'][:600]
            ))
            if len(documents) > 2:
                sections.append(Section(
                    title="Additional Details",
                    description=documents[2]['content'][:600]
                ))
        
        # Create the comprehensive guide
        guide = GuideOutline(
            title=f"{subject.title() if subject else 'Information'} Guide: {topic.title()}",
            introduction=f"This comprehensive guide provides detailed information about {topic} based on our knowledge base. The information has been curated to answer your query: '{query}'",
            target_audience=f"Professionals and enthusiasts interested in {subject or topic}",
            sections=sections,
            conclusion=f"This guide provided comprehensive information about {topic} from our specialized knowledge base covering {len(documents)} relevant sources."
        )
        
        result = guide.model_dump_json(indent=2)
        print(f"   ✅ Created comprehensive guide (length: {len(result)})")
        return result
        
    except Exception as e:
        print(f"   ⚠️ Error creating comprehensive guide: {e}")
        
        # Final fallback
        fallback_guide = GuideOutline(
            title=f"Information Guide: {topic}",
            introduction=f"This guide provides information about {query} focusing on {topic}",
            target_audience="General audience",
            sections=[
                Section(
                    title="Available Information",
                    description=combined_content[:1000] if combined_content else f"Information about {topic}"
                )
            ],
            conclusion=f"This guide provided available information about {topic} based on our knowledge base."
        )
        
        result = fallback_guide.model_dump_json(indent=2)
        print(f"   📄 Created fallback guide (length: {len(result)})")
        return result