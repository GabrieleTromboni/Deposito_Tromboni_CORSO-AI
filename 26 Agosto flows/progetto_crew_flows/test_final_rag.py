#!/usr/bin/env python3
"""
Test finale per il sistema RAG migliorato con la comunicazione corretta tra i tool.
"""

import os
import sys
import json
from pathlib import Path

# Aggiungi il percorso src al sys.path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew

def test_rag_system():
    """Test del sistema RAG migliorato con diversi tipi di query."""
    
    print("=" * 80)
    print("TEST FINALE DEL SISTEMA RAG MIGLIORATO")
    print("=" * 80)
    
    # Test query su diversi argomenti
    test_queries = [
        {
            "query": "What are the key characteristics of machine learning algorithms?",
            "topic": "machine learning",
            "subject": "technology"
        },
        {
            "query": "Come si sviluppa una strategia di marketing digitale efficace?",
            "topic": "marketing digitale",
            "subject": "business"
        },
        {
            "query": "Quali sono i principali benefici dell'allenamento cardiovascolare?",
            "topic": "allenamento cardiovascolare",
            "subject": "medicina"
        }
    ]
    
    for i, test_input in enumerate(test_queries, 1):
        print(f"\n--- TEST {i} ---")
        print(f"Query: {test_input['query']}")
        print(f"Topic: {test_input['topic']}")
        print(f"Subject: {test_input['subject']}")
        print("\nEsecuzione RAG crew...")
        
        try:
            crew = RAGCrew()
            result = crew.kickoff(inputs=test_input)
            
            print(f"\nRisultato tipo: {type(result)}")
            
            if hasattr(result, 'raw'):
                result_data = result.raw
            else:
                result_data = result
            
            # Verifica che il risultato sia un JSON strutturato
            if isinstance(result_data, str):
                try:
                    json_result = json.loads(result_data)
                    print("✅ Output JSON valido")
                    print(f"Titolo: {json_result.get('title', 'N/A')}")
                    print(f"Sezioni: {len(json_result.get('sections', []))}")
                except json.JSONDecodeError:
                    print("❌ Output non è JSON valido")
            elif hasattr(result_data, 'title'):
                print("✅ Output Pydantic valido")
                print(f"Titolo: {result_data.title}")
                print(f"Sezioni: {len(result_data.sections) if hasattr(result_data, 'sections') else 'N/A'}")
            else:
                print(f"⚠️ Tipo output imprevisto: {type(result_data)}")
            
        except Exception as e:
            print(f"❌ Errore durante il test {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETATI")
    print("=" * 80)

if __name__ == "__main__":
    test_rag_system()
