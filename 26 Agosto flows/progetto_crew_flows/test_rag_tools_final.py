#!/usr/bin/env python3
"""
Final validation test - Test complete RAG tool communication
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.tools.rag_tool import retrieve_from_vectordb, format_content_as_guide

def test_rag_tools_communication():
    """Test that the two RAG tools communicate correctly"""
    
    print("🧪 Testing RAG Tools Communication")
    print("=" * 60)
    
    # Test 1: Simple RAG workflow
    print("\n🔬 Test 1: Complete RAG workflow")
    
    try:
        # Step 1: Retrieve from vectordb
        print("   📥 Step 1: Retrieving from vector database...")
        retrieved_data = retrieve_from_vectordb._run(
            query="Spiegami i principi della cardiology",
            topic="cardiology", 
            subject="medicine",
            k=3
        )
        
        print(f"   ✅ Retrieved data length: {len(retrieved_data)}")
        print(f"   📄 Sample content: {retrieved_data[:200]}...")
        
        # Step 2: Format as guide
        print("   📝 Step 2: Formatting as guide...")
        guide_result = format_content_as_guide._run(
            retrieved_info=retrieved_data,
            query="Spiegami i principi della cardiology",
            topic="cardiology",
            subject="medicine"
        )
        
        print(f"   ✅ Guide created length: {len(guide_result)}")
        print(f"   📄 Guide preview: {guide_result[:300]}...")
        
        # Validate it's valid JSON
        import json
        guide_dict = json.loads(guide_result)
        print(f"   ✅ Valid JSON with keys: {list(guide_dict.keys())}")
        
        print("   🎯 SUCCESS: Tools communicate correctly!")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Error handling
    print("\n🔬 Test 2: Error handling for unknown topic")
    
    try:
        # Try with unknown topic
        retrieved_data = retrieve_from_vectordb._run(
            query="Tell me about quantum programming",
            topic="quantum programming",
            subject="physics",
            k=3
        )
        
        guide_result = format_content_as_guide._run(
            retrieved_info=retrieved_data,
            query="Tell me about quantum programming", 
            topic="quantum programming",
            subject="physics"
        )
        
        print(f"   ✅ Error handling works, guide length: {len(guide_result)}")
        
    except Exception as e:
        print(f"   ❌ ERROR in error handling: {e}")
    
    print("\n" + "=" * 60)
    print("🧪 RAG Tools Communication test completed")

if __name__ == "__main__":
    test_rag_tools_communication()
