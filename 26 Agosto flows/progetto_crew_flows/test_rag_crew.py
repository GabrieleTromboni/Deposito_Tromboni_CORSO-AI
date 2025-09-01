#!/usr/bin/env python3
"""Test script for the improved RAG crew tools"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew

def test_rag_crew():
    """Test the improved RAG crew with better chunking and communication"""
    print("\n" + "="*60)
    print("TESTING IMPROVED RAG CREW")
    print("="*60)
    
    # Initialize RAG crew
    rag_crew = RAGCrew()
    
    # Test inputs
    test_inputs = {
        "query": "What are the key differences between Premier League and Serie A?",
        "topic": "premier league",
        "subject": "football"
    }
    
    print(f"\nTest Query: {test_inputs['query']}")
    print(f"Topic: {test_inputs['topic']}")
    print(f"Subject: {test_inputs['subject']}")
    
    # Execute the crew
    try:
        print("\nExecuting RAG crew...")
        result = rag_crew.kickoff(inputs=test_inputs)
        
        print(f"\nResult type: {type(result)}")
        
        if hasattr(result, 'title'):
            print(f"Guide Title: {result.title}")
            print(f"Number of sections: {len(result.sections)}")
            print(f"Target audience: {result.target_audience}")
        else:
            print(f"Raw result: {str(result)[:500]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error executing RAG crew: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_crew()
    if success:
        print("\n✅ RAG crew test completed!")
    else:
        print("\n❌ RAG crew test failed!")
