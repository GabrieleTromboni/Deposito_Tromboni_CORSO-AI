#!/usr/bin/env python3
"""
Test routing behavior after architectural fix
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.progetto_crew_flows.WebRAG_flow import WebRAGFlow

def test_routing():
    """Test that routing works correctly based on SUBJECTS"""
    
    print("🧪 Testing WebRAGFlow routing behavior")
    print("=" * 60)
    
    flow = WebRAGFlow()
    
    # Test cases
    test_cases = [
        {
            "query": "Come posso migliorare la mia strategia di marketing digitale?",
            "expected_route": "WEB_SEARCH",
            "reason": "marketing digitale not in SUBJECTS"
        },
        {
            "query": "Quali sono i sintomi della cardiology moderna?",
            "expected_route": "RAG", 
            "reason": "cardiology is in SUBJECTS['medicine']"
        },
        {
            "query": "Spiegami l'artificial intelligence",
            "expected_route": "RAG",
            "reason": "artificial intelligence is in SUBJECTS['technology']"
        },
        {
            "query": "Come funziona la Premier League?",
            "expected_route": "RAG",
            "reason": "premier league is in SUBJECTS['football']"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {test_case['query']}")
        print(f"   Expected: {test_case['expected_route']} ({test_case['reason']})")
        
        try:
            # Initialize flow and set query
            flow.query_input = test_case['query']
            
            # Run extraction step
            state = flow.extraction()
            print(f"   📝 Extracted - Subject: {state.subject}, Topic: {state.topic}")
            
            # Run validation step
            state = flow.validation(state)
            print(f"   ✅ Validation - Subject valid: {state.is_valid_subject}, Topic valid: {state.is_valid_topic}")
            
            # Run router step
            route = flow.route_to_crew(state)
            print(f"   🎯 Route: {route}")
            
            # Check result
            if test_case['expected_route'] == "RAG" and route == "use_RAG":
                print(f"   ✅ PASS: Correctly routed to RAG")
            elif test_case['expected_route'] == "WEB_SEARCH" and route == "use_WEB_SEARCH":
                print(f"   ✅ PASS: Correctly routed to Web Search")
            else:
                print(f"   ❌ FAIL: Expected {test_case['expected_route']}, got {route}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("🧪 Routing test completed")

if __name__ == "__main__":
    test_routing()
