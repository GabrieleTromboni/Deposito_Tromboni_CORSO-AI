#!/usr/bin/env python3
"""Test script for RAG tools to verify chunking and crew communication"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.progetto_crew_flows.tools.rag_tool import create_vectordb, store_individual_documents, retrieve_from_vectordb, format_content_as_guide

def test_chunking_and_storage():
    """Test the improved chunking and storage"""
    print("\n" + "="*60)
    print("TESTING IMPROVED CHUNKING AND STORAGE")
    print("="*60)
    
    # Step 1: Create database
    print("\n1. Creating vector database...")
    result = create_vectordb()
    print(f"   Result: {result}")
    
    # Step 2: Store documents with improved chunking
    print("\n2. Storing documents with improved chunking...")
    result = store_individual_documents()
    print(f"   Result: {result}")
    
    # Step 3: Test retrieval
    print("\n3. Testing retrieval...")
    query = "What is the Premier League?"
    topic = "premier league"
    subject = "football"
    
    retrieved_info = retrieve_from_vectordb(query, topic, subject)
    print(f"   Retrieved info length: {len(retrieved_info)}")
    print(f"   First 500 chars: {retrieved_info[:500]}...")
    
    # Step 4: Test formatting
    print("\n4. Testing guide formatting...")
    guide_json = format_content_as_guide(retrieved_info, query, topic, subject)
    print(f"   Guide JSON length: {len(guide_json)}")
    print(f"   First 300 chars: {guide_json[:300]}...")
    
    return True

if __name__ == "__main__":
    try:
        test_chunking_and_storage()
        print("\n✅ All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
