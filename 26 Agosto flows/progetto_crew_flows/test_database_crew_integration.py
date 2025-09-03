"""
Test script for the new DatabaseCrew integration.

This script tests:
1. DatabaseCrew creation and configuration
2. Database creation (both FAISS and Qdrant)
3. RAG retrieval with different strategies
4. Integration with the updated WebRAG flow
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.crews.database_crew.database_crew import DatabaseCrew, create_database_crew
from src.progetto_crew_flows.WebRAG_flow import WebRAGFlow

def test_database_crew():
    """Test DatabaseCrew functionality"""
    print("🧪 Testing DatabaseCrew functionality...")
    
    # Test 1: Create DatabaseCrew instance
    print("\n1️⃣ Creating DatabaseCrew instance...")
    database_crew = create_database_crew()
    print("✅ DatabaseCrew created successfully")
    
    # Test 2: Test FAISS database creation
    print("\n2️⃣ Testing FAISS database creation...")
    try:
        result = database_crew.create_database(
            subject="medicine",
            topic="cardiology", 
            database_type="faiss"
        )
        print(f"✅ FAISS database creation result: {result}")
    except Exception as e:
        print(f"❌ FAISS database creation failed: {e}")
    
    # Test 3: Test Qdrant database creation (if available)
    print("\n3️⃣ Testing Qdrant database creation...")
    try:
        result = database_crew.create_database(
            subject="football",
            topic="premier league",
            database_type="qdrant",
            collection_name="football_premier_league"
        )
        print(f"✅ Qdrant database creation result: {result}")
    except Exception as e:
        print(f"❌ Qdrant database creation failed: {e}")
    
    # Test 4: Test RAG retrieval with FAISS
    print("\n4️⃣ Testing RAG retrieval with FAISS...")
    try:
        result = database_crew.execute_rag(
            query="Explain the principles of cardiology",
            subject="medicine",
            topic="cardiology",
            database_type="faiss",
            available_databases=["faiss"]
        )
        print(f"✅ FAISS RAG retrieval successful")
        if isinstance(result, dict) and 'title' in result.get('result', {}):
            print(f"   📖 Guide title: {result['result']['title']}")
        else:
            print(f"   📄 Result preview: {str(result)[:200]}...")
    except Exception as e:
        print(f"❌ FAISS RAG retrieval failed: {e}")
    
    # Test 5: Test RAG retrieval with Qdrant
    print("\n5️⃣ Testing RAG retrieval with Qdrant...")
    try:
        result = database_crew.execute_rag(
            query="Tell me about Premier League history",
            subject="football", 
            topic="premier league",
            database_type="qdrant",
            available_databases=["qdrant"]
        )
        print(f"✅ Qdrant RAG retrieval successful")
        if isinstance(result, dict) and 'title' in result.get('result', {}):
            print(f"   📖 Guide title: {result['result']['title']}")
        else:
            print(f"   📄 Result preview: {str(result)[:200]}...")
    except Exception as e:
        print(f"❌ Qdrant RAG retrieval failed: {e}")

def test_updated_flow():
    """Test the updated WebRAG flow with DatabaseCrew"""
    print("\n🌊 Testing updated WebRAG flow...")
    
    try:
        # Initialize flow
        flow = WebRAGFlow()
        
        # Test with a query that should trigger RAG
        print("\n6️⃣ Testing flow with RAG-compatible query...")
        result = flow.kickoff(inputs={"query": "Explain cardiology principles"})
        
        print(f"✅ Flow completed successfully")
        print(f"   📊 Subject: {result.subject}")
        print(f"   📊 Topic: {result.topic}")
        print(f"   📊 Source: {result.source_type}")
        print(f"   📊 Valid subject: {result.is_valid_subject}")
        print(f"   📊 Valid topic: {result.is_valid_topic}")
        
        if result.guide_outline:
            print(f"   📖 Guide title: {result.guide_outline.title}")
            print(f"   📖 Sections: {len(result.guide_outline.sections)}")
        
    except Exception as e:
        print(f"❌ Flow test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test execution"""
    print("🚀 Starting DatabaseCrew Integration Tests")
    print("=" * 60)
    
    # Test DatabaseCrew functionality
    test_database_crew()
    
    # Test updated flow
    test_updated_flow()
    
    print("\n" + "=" * 60)
    print("🏁 Tests completed!")

if __name__ == "__main__":
    main()
