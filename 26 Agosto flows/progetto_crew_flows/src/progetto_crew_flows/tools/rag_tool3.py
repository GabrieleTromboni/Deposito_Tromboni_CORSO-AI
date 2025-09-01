import os
import json
import time
import traceback
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from crewai.tools import tool
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from progetto_crew_flows.models import GuideOutline, Section

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
    chunk_size: int = 400  # Increased for richer content
    chunk_overlap: int = 50  # Increased proportionally
    search_type: str = "mmr"
    k: int = 3
    fetch_k: int = 10
    mmr_lambda: float = 0.4

SETTINGS = Settings()

# Helper functions for input normalization
def normalize_input(value):
    """Normalize input from CrewAI agents that might wrap strings in dict format"""
    if value is None:
        return None
    
    if isinstance(value, str):
        return value
    
    if isinstance(value, dict):
        # Handle CrewAI wrapped inputs like {'description': 'actual_value', 'type': 'str'}
        if 'description' in value:
            return str(value['description'])
        # Handle other dict structures
        if len(value) == 1:
            return str(list(value.values())[0])
        return str(value)
    
    return str(value)

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

@tool
def generate_documents(topics: List[str], docs_per_topic: int = 3, max_tokens_per_doc: int = 800, batch_size: int = 2, delay_between_batches: float = 2.0) -> str:
    """Generate documents for specified topics with detailed content for each topic"""
    
    MAX_EXECUTION_TIME = 300
    MAX_TOTAL_DOCUMENTS = 100
    
    start_time = time.time()
    
    if len(topics) * docs_per_topic > MAX_TOTAL_DOCUMENTS:
        print(f"⚠️ Warning: Requested {len(topics) * docs_per_topic} documents exceeds limit of {MAX_TOTAL_DOCUMENTS}")
        docs_per_topic = min(docs_per_topic, MAX_TOTAL_DOCUMENTS // len(topics))
    
    print(f"🚀 Starting document generation for {len(topics)} topics ({docs_per_topic} docs each)")

    llm = AzureChatOpenAI(
        deployment_name=CHAT_MODEL,
        openai_api_version=API_VERSION,
        azure_endpoint=CLIENT_AZURE,
        openai_api_key=API_KEY,
        temperature=0.2,
        max_tokens=max_tokens_per_doc,
        request_timeout=30
    )

    documents = []
    
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
        batch_topics = topics[batch_start:batch_start + batch_size]
        print(f"📦 Processing batch {batch_num + 1}/{total_batches}: {len(batch_topics)} topics")
        
        for topic in batch_topics:
            if time.time() - start_time > MAX_EXECUTION_TIME:
                print(f"⚠️ Maximum execution time ({MAX_EXECUTION_TIME}s) reached. Stopping generation.")
                break
            
            # Parse topic to extract subject and specific topic
            if " - " in topic:
                subject, specific_topic = topic.split(" - ", 1)
            else:
                subject = "general"
                specific_topic = topic
                
            print(f"   🔄 Generating {docs_per_topic} documents for: {subject} -> {specific_topic}")
            
            selected_strategies = doc_strategies[:docs_per_topic]
            
            for doc_num, strategy in enumerate(selected_strategies):
                try:
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", strategy["system_prompt"]),
                        ("human", strategy["user_prompt"].format(topic=specific_topic))
                    ])
                    
                    chain = prompt | llm
                    result = chain.invoke({"topic": specific_topic})
                    
                    content = result.content if hasattr(result, 'content') else str(result)
                    
                    # Create document with proper metadata for RAG retrieval
                    doc_info = {
                        "topic": specific_topic,
                        "subject": subject,
                        "content": content,
                        "source": f"generated_{subject}_{specific_topic}_{strategy['doc_type']}_doc{doc_num+1}",
                        "doc_type": strategy["doc_type"],
                        "focus_area": strategy["focus"],
                        "estimated_tokens": len(content.split()) * 1.3,
                        "generation_strategy": f"{strategy['focus']}_optimized",
                        "batch_number": batch_num + 1,
                        "full_topic": topic  # Keep original topic format for reference
                    }
                    
                    documents.append(doc_info)
                    print(f"     ✓ Generated doc {doc_num+1}: {strategy['doc_type']} for {specific_topic} ({len(content)} chars)")
                    
                except Exception as e:
                    print(f"     ✗ Error generating doc {doc_num+1} for {specific_topic}: {e}")
                    fallback_doc = {
                        "topic": specific_topic,
                        "subject": subject,
                        "content": f"Comprehensive information about {specific_topic}. This {subject} topic covers key aspects, definitions, applications, and current developments in the field. Important for understanding the broader context and specific details related to {specific_topic}.",
                        "source": f"fallback_{subject}_{specific_topic}_doc{doc_num+1}",
                        "doc_type": "fallback",
                        "focus_area": "basic",
                        "estimated_tokens": 50,
                        "generation_strategy": "fallback",
                        "batch_number": batch_num + 1,
                        "full_topic": topic
                    }
                    documents.append(fallback_doc)
        
        if batch_num < total_batches - 1:
            print(f"   ⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    total_tokens = sum(doc.get("estimated_tokens", 0) for doc in documents)
    generation_time = time.time() - start_time
    successful_docs = len([d for d in documents if d.get("doc_type") != "fallback"])
    
    print(f"\n📊 Generation Summary:")
    print(f"   ✓ {len(documents)} total documents ({successful_docs} successful, {len(documents)-successful_docs} fallback)")
    print(f"   ✓ Topics covered: {set([d.get('topic') for d in documents])}")
    print(f"   ✓ Subjects covered: {set([d.get('subject') for d in documents])}")
    print(f"   ✓ Total estimated tokens: {total_tokens:,}")
    print(f"   ✓ Generation time: {generation_time:.2f}s")
    
    return json.dumps(documents, ensure_ascii=False, indent=2)

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
def store_in_vectordb(content: str = None, **kwargs) -> str:
    """Store generated content in the FAISS vector database"""
    
    print(f"\n🔧 STORE_IN_VECTORDB DEBUG:")
    print(f"   Content parameter: {type(content)}")
    print(f"   Content preview: {str(content)[:200] if content else 'None'}...")
    print(f"   Additional kwargs: {list(kwargs.keys())}")
    
    # Priority: use explicit content parameter first
    actual_content = content
    topic = kwargs.get('topic')
    
    # If no explicit content, extract from kwargs
    if actual_content is None:
        if 'content' in kwargs:
            actual_content = kwargs['content']
            print(f"   Found content in kwargs: {type(actual_content)}")
        elif len(kwargs) == 1:
            key = list(kwargs.keys())[0]
            actual_content = kwargs[key]
            print(f"   Using single arg '{key}' as content: {type(actual_content)}")
        else:
            actual_content = kwargs
            print(f"   Using entire kwargs as content: {type(actual_content)}")
    
    # Handle the case where CrewAI wraps the data in a dict
    if isinstance(actual_content, dict):
        print(f"   Content is dict with keys: {list(actual_content.keys())}")
        
        # Check if it's a CrewAI-style dict with 'content' key
        if 'content' in actual_content:
            actual_content = actual_content['content']
            print(f"   Extracted 'content' from dict: {type(actual_content)}")
        
        # If it's still a dict, try to extract JSON string from keys/values
        elif len(actual_content) == 1:
            key, value = next(iter(actual_content.items()))
            if isinstance(key, str) and (key.startswith('[{') or key.startswith('{"')):
                actual_content = key
                print(f"   Using key as JSON content: {len(key)} characters")
            elif isinstance(value, str) and (value.startswith('[{') or value.startswith('{"')):
                actual_content = value
                print(f"   Using value as JSON content: {len(value)} characters")
        
        # Check for empty dict
        elif not actual_content:
            return "❌ Error: Empty content dict provided"
        if 'content' in actual_content:
            actual_content = actual_content['content']
            print(f"   Extracted content from dict, new type: {type(actual_content)}")
            if topic is None and 'topic' in kwargs:
                topic = kwargs['topic']
                print(f"   Extracted topic from kwargs: {topic}")
    
    if actual_content is None:
        return "❌ Error: No content provided to store"
    
    documents = []
    
    try:
        # Handle string input
        if isinstance(actual_content, str):
            content_str = actual_content.strip()
            print(f"   Processing as string, starts with: {content_str[:50]}")
            
            # Try to parse as JSON with multiple strategies
            if content_str.startswith('[') or content_str.startswith('{'):
                parsing_success = False
                
                # Strategy 1: Direct JSON parsing
                try:
                    parsed_content = json.loads(content_str)
                    print(f"   ✅ Successfully parsed JSON directly, type: {type(parsed_content)}")
                    parsing_success = True
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"   ❌ Direct JSON parsing failed: {e}")
                
                # Strategy 2: Handle double-escaped JSON (common with CrewAI)
                if not parsing_success and '\\\"' in content_str:
                    try:
                        # Try to decode escaped JSON
                        unescaped = content_str.encode().decode('unicode_escape')
                        parsed_content = json.loads(unescaped)
                        print(f"   ✅ Successfully parsed unescaped JSON, type: {type(parsed_content)}")
                        parsing_success = True
                    except Exception as e:
                        print(f"   ❌ Unescaped JSON parsing failed: {e}")
                
                # Strategy 3: Handle malformed JSON by fixing common issues
                if not parsing_success:
                    try:
                        # Fix common JSON issues like unterminated strings
                        fixed_content = content_str
                        
                        # Count opening and closing braces/brackets
                        open_braces = fixed_content.count('{')
                        close_braces = fixed_content.count('}')
                        open_brackets = fixed_content.count('[')
                        close_brackets = fixed_content.count(']')
                        
                        # Add missing closing characters
                        if open_brackets > close_brackets:
                            fixed_content += ']' * (open_brackets - close_brackets)
                        if open_braces > close_braces:
                            fixed_content += '}' * (open_braces - close_braces)
                        
                        parsed_content = json.loads(fixed_content)
                        print(f"   ✅ Successfully parsed fixed JSON, type: {type(parsed_content)}")
                        parsing_success = True
                    except Exception as e:
                        print(f"   ❌ Fixed JSON parsing failed: {e}")
                
                if parsing_success:
                    # Handle case where it's wrapped in another dict
                    if isinstance(parsed_content, dict) and 'content' in parsed_content:
                        parsed_content = parsed_content['content']
                        print(f"   Unwrapped content, new type: {type(parsed_content)}")
                    
                    # Handle list of documents
                    if isinstance(parsed_content, list):
                        print(f"   Processing {len(parsed_content)} documents from list")
                        for i, item in enumerate(parsed_content):
                            if isinstance(item, dict) and 'content' in item:
                                metadata = {
                                    "topic": str(item.get("topic", topic or "unknown")),
                                    "subject": str(item.get("subject", "general")),
                                    "source": str(item.get("source", "generated")),
                                    "doc_type": str(item.get("doc_type", "general")),
                                    "focus_area": str(item.get("focus_area", "general")),
                                    "generation_strategy": str(item.get("generation_strategy", "standard")),
                                    "full_topic": str(item.get("full_topic", item.get("topic", topic or "unknown")))
                                }
                                # Add additional metadata fields if present, converting to strings
                                for key, value in item.items():
                                    if key not in ["content", "topic", "subject", "source", "doc_type", "focus_area", "generation_strategy", "full_topic"] and value is not None:
                                        metadata[key] = str(value)
                                        
                                doc = Document(
                                    page_content=item.get("content", ""),
                                    metadata=metadata
                                )
                                documents.append(doc)
                                print(f"     ✓ Document {i+1}: subject='{metadata['subject']}', topic='{metadata['topic']}', type='{metadata['doc_type']}', length={len(item.get('content', ''))}")
                                
                    elif isinstance(parsed_content, dict) and 'content' in parsed_content:
                        # Single document in dict format
                        metadata = {
                            "topic": str(parsed_content.get("topic", topic or "unknown")),
                            "subject": str(parsed_content.get("subject", "general")),
                            "source": str(parsed_content.get("source", "generated")),
                            "doc_type": str(parsed_content.get("doc_type", "general")),
                            "focus_area": str(parsed_content.get("focus_area", "general")),
                            "generation_strategy": str(parsed_content.get("generation_strategy", "standard")),
                            "full_topic": str(parsed_content.get("full_topic", parsed_content.get("topic", topic or "unknown")))
                        }
                        doc = Document(
                            page_content=parsed_content.get("content", ""),
                            metadata=metadata
                        )
                        documents.append(doc)
                        print(f"     ✓ Single document: subject='{metadata['subject']}', topic='{metadata['topic']}', type='{metadata['doc_type']}'")
                else:
                    print(f"   ⚠️ All JSON parsing strategies failed, treating as plain text")
                # Plain text content
                doc = Document(
                    page_content=content_str,
                    metadata={
                        "topic": topic or "general", 
                        "subject": "general",
                        "source": f"text_{topic or 'content'}",
                        "doc_type": "plain_text",
                        "focus_area": "general",
                        "generation_strategy": "plain_text",
                        "full_topic": topic or "general"
                    }
                )
                documents.append(doc)
                print(f"     ✓ Plain text document: length={len(content_str)}")
        
        # Handle case where actual_content is already a parsed list/dict
        elif isinstance(actual_content, (list, dict)):
            print(f"   Content is already parsed: {type(actual_content)}")
            parsed_content = actual_content
            
            if isinstance(parsed_content, list):
                for i, item in enumerate(parsed_content):
                    if isinstance(item, dict) and 'content' in item:
                        metadata = {
                            "topic": str(item.get("topic", topic or "unknown")),
                            "subject": str(item.get("subject", "general")),
                            "source": str(item.get("source", "generated")),
                            "doc_type": str(item.get("doc_type", "general")),
                            "focus_area": str(item.get("focus_area", "general")),
                            "generation_strategy": str(item.get("generation_strategy", "standard")),
                            "full_topic": str(item.get("full_topic", item.get("topic", topic or "unknown")))
                        }
                        doc = Document(
                            page_content=item.get("content", ""),
                            metadata=metadata
                        )
                        documents.append(doc)
                        print(f"     ✓ Document {i+1}: subject='{metadata['subject']}', topic='{metadata['topic']}', type='{metadata['doc_type']}'")
                        
            elif isinstance(parsed_content, dict) and 'content' in parsed_content:
                metadata = {
                    "topic": str(parsed_content.get("topic", topic or "unknown")),
                    "subject": str(parsed_content.get("subject", "general")),
                    "source": str(parsed_content.get("source", "generated")),
                    "doc_type": str(parsed_content.get("doc_type", "general")),
                    "focus_area": str(parsed_content.get("focus_area", "general")),
                    "generation_strategy": str(parsed_content.get("generation_strategy", "standard")),
                    "full_topic": str(parsed_content.get("full_topic", parsed_content.get("topic", topic or "unknown")))
                }
                doc = Document(
                    page_content=parsed_content.get("content", ""),
                    metadata=metadata
                )
                documents.append(doc)
                print(f"     ✓ Single document: subject='{metadata['subject']}', topic='{metadata['topic']}', type='{metadata['doc_type']}'")
                
    except (json.JSONDecodeError, ValueError) as e:
        print(f"   JSON parsing failed: {e}")
        print(f"   Treating as plain text content")
        pass
    
    # If no documents were parsed from JSON, treat as plain text
    if not documents:
        print(f"   Creating document from plain text (length: {len(str(actual_content))})")
        doc = Document(
            page_content=str(actual_content),
            metadata={
                "topic": topic or "general", 
                "subject": "general",
                "source": f"generated_{topic or 'content'}.md",
                "doc_type": "plain_text",
                "focus_area": "general",
                "generation_strategy": "plain_text",
                "full_topic": topic or "general"
            }
        )
        documents.append(doc)
    
    if not documents:
        return "❌ Error: No valid documents found to store"
    
    print(f"   📄 Total documents to process: {len(documents)}")
    
    # Group documents by subject and topic for better organization
    topic_distribution = {}
    for doc in documents:
        subject = doc.metadata.get("subject", "general")
        doc_topic = doc.metadata.get("topic", "unknown")
        key = f"{subject} -> {doc_topic}"
        topic_distribution[key] = topic_distribution.get(key, 0) + 1
    
    print(f"   📊 Document distribution:")
    for key, count in topic_distribution.items():
        print(f"     {key}: {count} documents")
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"   🔪 Split into {len(chunks)} chunks")
    
    # Load or create vector store
    persist_dir = SETTINGS.persist_dir
    try:
        if Path(persist_dir).exists() and Path(persist_dir, "index.faiss").exists():
            print(f"   📚 Loading existing vector store from {persist_dir}")
            vector_store = FAISS.load_local(
                persist_dir,
                EMBEDDINGS,
                allow_dangerous_deserialization=True
            )
            vector_store.add_documents(chunks)
            print(f"   ➕ Added {len(chunks)} chunks to existing store")
        else:
            print(f"   🆕 Creating new vector store at {persist_dir}")
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            vector_store = FAISS.from_documents(chunks, EMBEDDINGS)
            print(f"   ✅ Created new store with {len(chunks)} chunks")

        # Save vector store
        vector_store.save_local(persist_dir)
        print(f"   💾 Saved vector store successfully")
        
        # Extract unique topics and subjects stored
        unique_topics = set([doc.metadata.get("topic", "unknown") for doc in documents])
        unique_subjects = set([doc.metadata.get("subject", "unknown") for doc in documents])
        
        result = f"✅ Successfully stored {len(chunks)} chunks from {len(documents)} documents.\n"
        result += f"   Subjects: {', '.join(sorted(unique_subjects))}\n"
        result += f"   Topics: {', '.join(sorted(unique_topics))}\n"
        result += f"   Distribution: {dict(topic_distribution)}"
        
        print(f"   🎉 {result}")
        return result
    
    except Exception as e:
        error_msg = f"❌ Error storing documents in vector database: {str(e)}"
        print(f"   {error_msg}")
        traceback.print_exc()
        return error_msg

@tool
def retrieve_from_vectordb(query: str, topic: Optional[str] = None, subject: Optional[str] = None, k: Optional[int] = None) -> str:
    """Retrieve information from the vector database - SINGLE EXECUTION ONLY"""
    
    print(f"\n🔍 RETRIEVE_FROM_VECTORDB - SINGLE EXECUTION:")
    
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
        error_msg = "Empty or invalid query provided"
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "Invalid query", "content": error_msg, "final": True}])
    
    # Check embeddings availability
    if not safe_embeddings_check():
        error_msg = "Azure OpenAI embeddings are not available. Connectivity issue detected."
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "Embeddings unavailable", "content": error_msg, "final": True}])
    
    # Load FAISS index from correct path
    persist_dir = SETTINGS.persist_dir
    print(f"   Persist dir: {persist_dir}")
    print(f"   Dir exists: {Path(persist_dir).exists()}")
    print(f"   Index exists: {Path(persist_dir, 'index.faiss').exists()}")
    
    if not Path(persist_dir).exists() or not Path(persist_dir, "index.faiss").exists():
        error_msg = f"Vector database not found at {persist_dir}. Please ensure RAG database is properly initialized."
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "Vector database not found", "content": error_msg, "final": True}])
    
    try:
        vector_store = FAISS.load_local(
            persist_dir,
            EMBEDDINGS,
            allow_dangerous_deserialization=True
        )
        print(f"   ✅ Vector store loaded successfully")
        
        # Check the total number of documents in the vector store
        total_docs = vector_store.index.ntotal
        print(f"   📊 Total documents in vector store: {total_docs}")
        
    except Exception as e:
        error_msg = f"Error loading vector store: {str(e)}"
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "Database load error", "content": error_msg, "final": True}])
    
    # Build search query with both subject and topic if available
    if subject and topic:
        search_query = f"{subject} {topic}: {query}"
    elif topic:
        search_query = f"{topic}: {query}"
    else:
        search_query = query
    
    print(f"   🔎 Search query: '{search_query}'")
    
    # Use custom k if provided (default to 5 for focused results)
    try:
        search_k = int(k) if k and str(k).isdigit() else 5
    except (ValueError, TypeError):
        search_k = 5
    print(f"   📝 Search parameters: k={search_k}")
    
    try:
        # Try multiple search strategies with error handling
        docs = []
        search_attempts = [
            ("full query", search_query),
            ("topic only", topic if topic else ""),
            ("subject only", subject if subject else ""),
            ("original query", query)
        ]
        
        for attempt_name, search_term in search_attempts:
            if not search_term or search_term.strip() == "":
                continue
                
            try:
                print(f"   🔄 Trying {attempt_name}: '{search_term}'")
                attempt_docs = vector_store.similarity_search(search_term, k=search_k)
                if attempt_docs:
                    docs.extend(attempt_docs)
                    print(f"   📄 {attempt_name} found {len(attempt_docs)} documents")
                    break  # Stop at first successful search
            except Exception as search_error:
                print(f"   ⚠️ {attempt_name} search failed: {search_error}")
                continue
        
        if not docs:
            # Final fallback: try with just keywords
            query_words = query.lower().split()
            for word in query_words[:3]:  # Try only first 3 words to avoid too many searches
                if len(word) > 3:  # Only try meaningful words
                    try:
                        word_docs = vector_store.similarity_search(word, k=2)
                        if word_docs:
                            docs.extend(word_docs)
                            print(f"   📄 Word '{word}' search found {len(word_docs)} documents")
                            if len(docs) >= 3:
                                break
                    except Exception as word_error:
                        print(f"   ⚠️ Word '{word}' search failed: {word_error}")
                        continue
        
        # If no results with complex query, try simpler searches
        if not docs and (subject or topic):
            print(f"   🔄 Trying simpler search strategies...")
            
            # Try searching for just the topic
            if topic:
                simple_docs = vector_store.similarity_search(topic, k=search_k)
                print(f"   📄 Topic-only search found {len(simple_docs)} documents")
                docs.extend(simple_docs)
            
            # Try searching for just the subject
            if subject and not docs:
                subject_docs = vector_store.similarity_search(subject, k=search_k)
                print(f"   📄 Subject-only search found {len(subject_docs)} documents")
                docs.extend(subject_docs)
            
            # Try searching for individual keywords from the query
            if not docs:
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 3:  # Only try meaningful words
                        word_docs = vector_store.similarity_search(word, k=2)
                        print(f"   📄 Word '{word}' search found {len(word_docs)} documents")
                        docs.extend(word_docs)
                        if len(docs) >= 3:  # Limit to avoid too many irrelevant results
                            break
                            
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "Search error", "content": error_msg, "final": True}])
    
    # Remove duplicates while preserving order
    seen_content = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)
    
    docs = unique_docs[:search_k]  # Limit to requested number
    print(f"   🔹 Final unique documents: {len(docs)}")
    
    # Return raw content for processing
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        doc_topic = doc.metadata.get("topic", "general")
        doc_subject = doc.metadata.get("subject", "general")
        content = doc.page_content
        
        print(f"     Doc {i+1}: topic='{doc_topic}', subject='{doc_subject}', source='{source}', length={len(content)}")
        
        results.append({
            "content": content,
            "topic": doc_topic,
            "subject": doc_subject,
            "source": source,
            "final": True  # Mark as final to prevent re-execution
        })
    
    if not results:
        error_msg = f"No relevant information found for query: {search_query}"
        print(f"   ❌ {error_msg}")
        return json.dumps([{"error": "No results", "content": error_msg, "final": True}])
    
    print(f"   ✅ Returning {len(results)} results - EXECUTION COMPLETE")
    # Return as JSON string for easy parsing
    return json.dumps(results, ensure_ascii=False, indent=2)

@tool
def format_content_as_guide(retrieved_info: str, query: str, topic: str, subject: Optional[str] = None) -> str:
    """Format retrieved content into a structured guide - SINGLE EXECUTION ONLY"""
    
    print(f"\n📝 FORMAT_CONTENT_AS_GUIDE - SINGLE EXECUTION:")
    
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
    if not retrieved_info:
        retrieved_info = "No information available"
    
    # Parse retrieved information
    try:
        docs = json.loads(retrieved_info) if isinstance(retrieved_info, str) else retrieved_info
        print(f"   ✅ Successfully parsed retrieved info as JSON")
        
        # Check for errors in retrieved data
        if docs and isinstance(docs[0], dict) and "error" in docs[0]:
            print(f"   ⚠️ Error detected in retrieved data: {docs[0].get('error')}")
            # Create minimal guide with error info instead of failing
            error_guide = GuideOutline(
                title=f"Information Request: {topic}",
                introduction=f"We attempted to find information about {topic} in our database.",
                target_audience="General audience",
                sections=[
                    Section(
                        title="Search Status",
                        description=f"Database search completed but no specific information was found for '{topic}'. This may indicate the topic is not yet covered in our knowledge base."
                    ),
                    Section(
                        title="Alternative Approach",
                        description="For comprehensive information about this topic, consider using web search or consulting external sources."
                    )
                ],
                conclusion="The requested information was not available in our current knowledge base."
            )
            result = error_guide.model_dump_json(indent=2)
            print(f"   📄 Created error guide (length: {len(result)}) - EXECUTION COMPLETE")
            return result
            
    except Exception as e:
        print(f"   ⚠️ JSON parsing failed: {e}, treating as plain text")
        docs = [{"content": retrieved_info, "topic": topic, "source": "unknown", "final": True}]
    
    # Combine all content
    combined_content = "\n\n".join([doc.get("content", "") for doc in docs if "error" not in doc])
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
                    title="Context", 
                    description=f"Your query was: '{query}'. This topic may require additional research from external sources."
                ),
                Section(
                    title="Next Steps", 
                    description="Consider refining your search terms or exploring related topics that may be available in our knowledge base."
                )
            ],
            conclusion="This guide represents the available information for your query. For more comprehensive details, additional sources may be needed."
        )
        result = minimal_guide.model_dump_json(indent=2)
        print(f"   📄 Created minimal guide (length: {len(result)}) - EXECUTION COMPLETE")
        return result
    
    print(f"   🚀 Creating comprehensive guide from content")
    
    # Create a comprehensive guide with the available content
    try:
        # Split content into logical sections
        content_sections = []
        content_parts = combined_content.split('\n\n')
        
        if len(content_parts) >= 3:
            content_sections = [
                Section(title="Overview", description=content_parts[0][:800]),
                Section(title="Key Information", description=content_parts[1][:800] if len(content_parts) > 1 else "Additional information about the topic"),
                Section(title="Details", description=content_parts[2][:800] if len(content_parts) > 2 else "Further details and applications")
            ]
        else:
            # If content is limited, create sections from the full text
            full_text = combined_content
            sections_count = min(3, max(1, len(full_text) // 500))
            
            if sections_count == 1:
                content_sections = [
                    Section(title="Information", description=full_text[:1000])
                ]
            elif sections_count == 2:
                mid_point = len(full_text) // 2
                content_sections = [
                    Section(title="Overview", description=full_text[:mid_point]),
                    Section(title="Details", description=full_text[mid_point:])
                ]
            else:
                third = len(full_text) // 3
                content_sections = [
                    Section(title="Overview", description=full_text[:third]),
                    Section(title="Key Information", description=full_text[third:2*third]),
                    Section(title="Additional Details", description=full_text[2*third:])
                ]
        
        # Create comprehensive guide
        comprehensive_guide = GuideOutline(
            title=f"{subject.title() if subject else 'Information'} Guide: {topic.title()}",
            introduction=f"This comprehensive guide provides detailed information about {topic} based on our knowledge base. The information has been curated to answer your query: '{query}'",
            target_audience=f"Professionals and enthusiasts interested in {subject or topic}",
            sections=content_sections,
            conclusion=f"This guide provided comprehensive information about {topic} from our specialized knowledge base. The content covers key aspects relevant to your inquiry."
        )
        
        final_result = comprehensive_guide.model_dump_json(indent=2)
        print(f"   ✅ Created comprehensive guide (length: {len(final_result)}) - EXECUTION COMPLETE")
        return final_result
        
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
        
        final_result = fallback_guide.model_dump_json(indent=2)
        print(f"   📄 Created fallback guide (length: {len(final_result)}) - EXECUTION COMPLETE")
        return final_result
