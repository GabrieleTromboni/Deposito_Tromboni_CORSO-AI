#!/usr/bin/env python3
"""
Test complete end-to-end flow after architectural fixes
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.WebRAG_flow import WebRAGFlow

def test_end_to_end():
    """Test complete flow execution"""
    
    print("🧪 Testing complete end-to-end flow")
    print("=" * 60)
    
    test_cases = [
        {
            "query": "Come posso migliorare la mia strategia di marketing digitale?",
            "expected_source": "WEB_SEARCH"
        },
        {
            "query": "Spiegami i principi della cardiology",
            "expected_source": "RAG"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 End-to-End Test {i}: {test_case['query']}")
        print(f"   Expected source: {test_case['expected_source']}")
        
        try:
            flow = WebRAGFlow()
            
            # Run complete flow
            result = flow.kickoff(inputs={"query": test_case['query']})
            
            print(f"   🎯 Flow completed successfully")
            print(f"   📊 Result type: {type(result)}")
            
            # Extract source type from result if available
            if hasattr(result, 'source_type'):
                actual_source = result.source_type
                print(f"   📝 Actual source: {actual_source}")
                
                if test_case['expected_source'] == actual_source:
                    print(f"   ✅ PASS: Correct source used")
                else:
                    print(f"   ❌ FAIL: Expected {test_case['expected_source']}, got {actual_source}")
            else:
                print(f"   ⚠️ Could not determine source type from result")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🧪 End-to-end test completed")

if __name__ == "__main__":
    test_end_to_end()
