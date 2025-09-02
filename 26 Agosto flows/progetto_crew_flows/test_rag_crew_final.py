#!/usr/bin/env python3
"""
Test complete RAG crew workflow
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew

def test_rag_crew():
    """Test complete RAG crew workflow"""
    
    print("🧪 Testing RAG Crew Complete Workflow")
    print("=" * 60)
    
    test_cases = [
        {
            "query": "Spiegami i principi della cardiology",
            "topic": "cardiology",
            "subject": "medicine"
        },
        {
            "query": "Tell me about Premier League",
            "topic": "premier league", 
            "subject": "football"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {test_case['query']}")
        print(f"   Subject: {test_case['subject']}, Topic: {test_case['topic']}")
        
        try:
            crew = RAGCrew()
            
            print("   🚀 Starting crew execution...")
            result = crew.kickoff(inputs=test_case)
            
            print(f"   ✅ Crew completed successfully!")
            print(f"   📊 Result type: {type(result)}")
            
            if hasattr(result, 'title'):
                print(f"   📖 Guide title: {result.title}")
                print(f"   🎯 Sections: {len(result.sections) if hasattr(result, 'sections') else 0}")
            else:
                print(f"   📄 Result preview: {str(result)[:200]}...")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🧪 RAG Crew test completed")

if __name__ == "__main__":
    test_rag_crew()
