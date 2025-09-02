#!/usr/bin/env python3
"""
Test integration of enhanced Qdrant tools with RAG Crew
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.tools.rag_tool import (
    intelligent_rag_search,
    qdrant_hybrid_search,
    qdrant_semantic_search,
    qdrant_text_search,
    SETTINGS
)
from src.progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew
import json

def test_database_detection():
    """Test automatic database detection"""
    print("🔍 Testing Database Detection")
    print("=" * 50)
    
    crew = RAGCrew()
    available_dbs = crew._detect_available_databases()
    
    print(f"Available databases: {available_dbs}")
    print(f"Qdrant URL: {SETTINGS.qdrant_url}")
    print(f"Qdrant Collection: {SETTINGS.persist_dir}")
    
    return available_dbs

def test_individual_qdrant_tools():
    """Test each Qdrant tool individually"""
    print("\n🔧 Testing Individual Qdrant Tools")
    print("=" * 50)
    
    test_queries = [
        {
            "query": "What is cardiology?",
            "topic": "cardiology",
            "subject": "medicine"
        },
        {
            "query": "Explain Premier League",
            "topic": "premier league",
            "subject": "football"
        }
    ]
    
    tools_to_test = [
        ("Intelligent RAG Search", intelligent_rag_search),
        ("Hybrid Search", qdrant_hybrid_search),
        ("Semantic Search", qdrant_semantic_search),
        ("Text Search", qdrant_text_search)
    ]
    
    for query_data in test_queries:
        print(f"\n📝 Query: {query_data['query']}")
        print(f"   Subject: {query_data['subject']}, Topic: {query_data['topic']}")
        
        for tool_name, tool_func in tools_to_test:
            print(f"\n   🔧 Testing {tool_name}...")
            try:
                if tool_name == "Hybrid Search" or tool_name == "Intelligent RAG Search":
                    result = tool_func(
                        query=query_data["query"],
                        k=3,
                        topic=query_data["topic"],
                        subject=query_data["subject"]
                    )
                else:
                    result = tool_func(
                        query=query_data["query"],
                        limit=5,
                        topic=query_data["topic"],
                        subject=query_data["subject"]
                    )
                
                # Parse and analyze result
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "success":
                        print(f"      ✅ Success: {len(result_data.get('documents', []))} documents")
                        if "selected_strategy" in result_data:
                            print(f"      🎯 Strategy: {result_data['selected_strategy']}")
                        if "search_type" in result_data:
                            print(f"      🔍 Type: {result_data['search_type']}")
                    elif result_data.get("status") == "no_results":
                        print(f"      ⚠️ No results found")
                    else:
                        print(f"      ❌ Error: {result_data.get('message', 'Unknown error')}")
                except json.JSONDecodeError:
                    if "ERROR:" in result:
                        print(f"      ❌ Error: {result}")
                    else:
                        print(f"      ⚠️ Non-JSON result (length: {len(result)})")
                        
            except Exception as e:
                print(f"      ❌ Exception: {e}")

def test_strategy_selection():
    """Test intelligent strategy selection"""
    print("\n🧠 Testing Strategy Selection")
    print("=" * 50)
    
    strategy_test_queries = [
        # Should select semantic search
        ("Explain the concept of machine learning", "semantic"),
        ("How does photosynthesis work?", "semantic"),
        ("What is the relationship between AI and ML?", "semantic"),
        
        # Should select text search
        ("What is the definition of API?", "text"),
        ("Who is the CEO of Tesla?", "text"),
        ("When was Python created?", "text"),
        
        # Should select hybrid search
        ("Comprehensive guide to football tactics", "hybrid"),
        ("Detailed analysis of cardiovascular diseases", "hybrid"),
        ("Overview of modern programming languages", "hybrid")
    ]
    
    for query, expected_strategy in strategy_test_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Expected strategy: {expected_strategy}")
        
        try:
            result = intelligent_rag_search(query=query, k=3)
            result_data = json.loads(result)
            
            if "selected_strategy" in result_data:
                actual_strategy = result_data["selected_strategy"]
                status = "✅" if actual_strategy == expected_strategy else "⚠️"
                print(f"   {status} Selected: {actual_strategy}")
                if "strategy_reasoning" in result_data:
                    print(f"   💭 Reasoning: {result_data['strategy_reasoning'][:100]}...")
            else:
                print(f"   ❌ No strategy selection found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_rag_crew_integration():
    """Test RAG Crew with enhanced tools"""
    print("\n🚀 Testing RAG Crew Integration")
    print("=" * 50)
    
    test_cases = [
        {
            "query": "Explain the principles of cardiology and heart health",
            "topic": "cardiology",
            "subject": "medicine"
        },
        {
            "query": "Tell me about Premier League football history",
            "topic": "premier league",
            "subject": "football"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {test_case['query']}")
        print(f"   Subject: {test_case['subject']}, Topic: {test_case['topic']}")
        
        try:
            crew = RAGCrew()
            result = crew.kickoff(inputs=test_case)
            
            print(f"   ✅ Crew completed successfully!")
            print(f"   📊 Result type: {type(result)}")
            
            if hasattr(result, 'title'):
                print(f"   📖 Guide title: {result.title}")
                print(f"   🎯 Sections: {len(result.sections) if hasattr(result, 'sections') else 0}")
                if hasattr(result, 'sections') and result.sections:
                    print(f"   📝 First section: {result.sections[0].title}")
            else:
                print(f"   📄 Result preview: {str(result)[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Qdrant Tools Integration")
    print("=" * 60)
    
    try:
        # Test 1: Database detection
        available_dbs = test_database_detection()
        
        if not available_dbs:
            print("\n⚠️ No databases available. Please check configuration.")
            print("   - Qdrant: Check QDRANT_URL and QDRANT_COLLECTION env vars")
            print("   - FAISS: Check if vector database has been created")
            return
        
        # Test 2: Individual tools (only if databases available)
        if "qdrant" in available_dbs:
            test_individual_qdrant_tools()
            test_strategy_selection()
        else:
            print("\n⚠️ Qdrant not available, skipping Qdrant-specific tests")
        
        # Test 3: RAG Crew integration
        test_rag_crew_integration()
        
        print("\n✅ All tests completed!")
        print(f"📊 Available databases: {available_dbs}")
        
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
