#!/usr/bin/env python3
"""
Test specifico per verificare il web search per marketing digitale
"""

import os
import sys
import json
from pathlib import Path

# Aggiungi il percorso src al sys.path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew

def test_marketing_query():
    """Test specifico per la query di marketing digitale che dovrebbe attivare web search"""
    
    print("=" * 80)
    print("TEST WEB SEARCH - MARKETING DIGITALE")
    print("=" * 80)
    
    # Query che dovrebbe attivare web search
    test_input = {
        "query": "Come si sviluppa una strategia di marketing digitale efficace?",
        "topic": "marketing digitale", 
        "subject": "business"
    }
    
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
                
                # Verifica se contiene informazioni da web search
                content_str = json.dumps(json_result, ensure_ascii=False).lower()
                if 'web search' in content_str or 'web_search' in content_str:
                    print("🌐 ✅ CONTENUTO DA WEB SEARCH RILEVATO!")
                else:
                    print("⚠️ Nessun contenuto da web search rilevato")
                    
            except json.JSONDecodeError:
                print("❌ Output non è JSON valido")
        elif hasattr(result_data, 'title'):
            print("✅ Output Pydantic valido")
            print(f"Titolo: {result_data.title}")
            print(f"Sezioni: {len(result_data.sections) if hasattr(result_data, 'sections') else 'N/A'}")
        else:
            print(f"⚠️ Tipo output imprevisto: {type(result_data)}")
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETATO")
    print("=" * 80)

if __name__ == "__main__":
    test_marketing_query()
