#!/usr/bin/env python3
"""
Test rapido per verificare la dimensione degli embeddings Azure OpenAI
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
load_dotenv()

def test_embedding_dimensions():
    """Test Azure OpenAI embedding dimensions"""
    
    print("🔍 Testing Azure OpenAI Embedding Dimensions")
    print("="*50)
    
    try:
        from langchain_openai import AzureOpenAIEmbeddings
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )
        
        print("✅ Azure OpenAI embeddings initialized")
        
        # Test embedding
        test_text = "This is a test for vector dimensions"
        embedding = embeddings.embed_query(test_text)
        
        print(f"📏 Embedding dimensions: {len(embedding)}")
        print(f"🔢 Sample values (first 5): {embedding[:5]}")
        
        return len(embedding)
        
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return None

if __name__ == "__main__":
    dimension = test_embedding_dimensions()
    
    if dimension:
        print(f"\n✅ SUCCESS: Azure OpenAI embeddings dimension = {dimension}")
        print(f"💡 Use this dimension for Qdrant collection creation")
    else:
        print(f"\n❌ FAILED: Could not determine embedding dimensions")
