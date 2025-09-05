"""
Esempio di integrazione del sistema di valutazione con WebRAG Flow
Dimostra come aggiungere valutazione automatica al tuo flow esistente
"""

# 1. INTEGRAZIONE RAPIDA - Aggiungi queste righe al tuo WebRAG_flow.py
"""
from .evaluation import auto_evaluate, FlowEvaluationMixin

# Opzione A: Usa il decorator per valutazione automatica
class WebRAG_flow(FlowEvaluationMixin, Flow):
    @auto_evaluate(source_type="RAG")
    def use_RAG(self, query, subject, topic):
        # Il tuo codice esistente...
        pass
    
    @auto_evaluate(source_type="Web Search")  
    def use_web_search(self, query, subject, topic):
        # Il tuo codice esistente...
        pass
"""

# 2. ESEMPIO COMPLETO - Copia e modifica questo file per testare
from crewai import Flow
from typing import Dict, Any
import sys
from pathlib import Path

# Aggiungi path del modulo evaluation
sys.path.append(str(Path(__file__).parent / "src" / "progetto_crew_flows"))

from evaluation import (
    CrewEvaluator, 
    auto_evaluate, 
    FlowEvaluationMixin,
    EvaluationDashboard,
    quick_evaluate
)

class EvaluatedWebRAGFlow(FlowEvaluationMixin, Flow):
    """
    Versione del WebRAG Flow con valutazione automatica integrata
    """
    
    def __init__(self):
        super().__init__()
        
        # Configura valutatore (opzionale)
        self.set_evaluation_config({
            'enable_quality_eval': True,
            'enable_performance_eval': True,
            'enable_cost_eval': True,
            'enable_accuracy_eval': True,
            'save_results': True,
            'detailed_logging': True
        })
    
    @Flow.listen("flow_start")
    def extraction(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Estrai subject e topic dalla query"""
        # Simulazione estrazione (sostituisci con il tuo codice)
        return {
            "query": query.get("query", ""),
            "subject": "technology",  # Tuo logic di estrazione
            "topic": "AI"             # Tuo logic di estrazione
        }
    
    @Flow.listen("extraction") 
    def validation(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida se il topic è nel database"""
        # Simulazione validazione
        return {
            **query_data,
            "is_valid": True  # Tua logica di validazione
        }
    
    @Flow.listen("validation")
    def route_to_crew(self, validated_data: Dict[str, Any]) -> str:
        """Routing tra RAG e Web Search"""
        if validated_data.get("is_valid", False):
            return "use_RAG"
        else:
            return "use_web_search"
    
    @Flow.listen("route_to_crew", condition=lambda result: result == "use_RAG")
    @auto_evaluate(source_type="RAG Database")
    def use_RAG(self, query: str, subject: str, topic: str) -> Dict[str, Any]:
        """
        Usa RAG crew con valutazione automatica
        """
        print(f"🔍 Executing RAG for: {subject}/{topic}")
        
        # Simulazione del tuo RAG crew (sostituisci con il tuo codice)
        # result = self.database_crew.kickoff(query, subject, topic)
        
        # Simulazione risultato
        result = {
            "source_type": "RAG Database",
            "guide_outline": type('obj', (object,), {
                'title': f"Guida {topic} per {subject}",
                'introduction': f"Introduzione completa a {topic} nel contesto di {subject}",
                'sections': [
                    type('section', (object,), {
                        'title': f"Fondamenti di {topic}",
                        'description': f"Concetti base e principi fondamentali di {topic}"
                    })(),
                    type('section', (object,), {
                        'title': f"Applicazioni pratiche",
                        'description': f"Come applicare {topic} in scenari reali"
                    })()
                ],
                'conclusion': f"Riassunto e prossimi passi per {topic}",
                'target_audience': "Principianti e intermedi"
            })(),
            'sources': ['Database interno', 'Documentazione RAG'],
            'confidence': 0.95
        }
        
        return result
    
    @Flow.listen("route_to_crew", condition=lambda result: result == "use_web_search")
    @auto_evaluate(source_type="Web Search")  
    def use_web_search(self, query: str, subject: str, topic: str) -> Dict[str, Any]:
        """
        Usa Web Search crew con valutazione automatica
        """
        print(f"🌐 Executing Web Search for: {subject}/{topic}")
        
        # Simulazione del tuo web search crew
        result = {
            "source_type": "Web Search",
            "guide_outline": type('obj', (object,), {
                'title': f"Guida Web: {topic} in {subject}",
                'introduction': f"Informazioni aggiornate su {topic} trovate sul web",
                'sections': [
                    type('section', (object,), {
                        'title': f"Novità su {topic}",
                        'description': f"Ultime novità e trend per {topic}"
                    })(),
                    type('section', (object,), {
                        'title': f"Risorse online",
                        'description': f"Migliori risorse web per {topic}"
                    })()
                ],
                'conclusion': f"Riepilogo delle informazioni web su {topic}",
                'target_audience': "Tutti i livelli"
            })(),
            'sources': ['Wikipedia', 'GitHub', 'Stack Overflow'],
            'confidence': 0.80
        }
        
        return result

def demo_evaluation_system():
    """
    Demo completa del sistema di valutazione
    """
    print("🚀 DEMO SISTEMA DI VALUTAZIONE CREWAI FLOWS")
    print("=" * 60)
    
    # 1. Crea flow con valutazione
    flow = EvaluatedWebRAGFlow()
    
    # 2. Esegui alcune query di test
    test_queries = [
        {
            "query": "Come implementare machine learning in Python?",
            "subject": "programming",
            "topic": "machine_learning"
        },
        {
            "query": "Spiegami i database NoSQL",
            "subject": "database", 
            "topic": "nosql"
        },
        {
            "query": "Tutorial Docker per principianti",
            "subject": "devops",
            "topic": "docker"
        }
    ]
    
    print("\n📊 Esecuzione query di test...")
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/3 ---")
        
        try:
            # Kickoff del flow (automaticamente valutato)
            result = flow.kickoff(test_query)
            
            # Ottieni ultima valutazione
            evaluation = flow.evaluate_last_execution()
            if evaluation:
                print(f"✅ Score: {evaluation.overall_score:.3f} (Grado: {evaluation.grade})")
                print(f"   - Qualità: {evaluation.quality_metrics.overall_quality:.3f}")
                print(f"   - Accuratezza: {evaluation.accuracy_metrics.overall_accuracy:.3f}")
                print(f"   - Durata: {evaluation.performance_metrics.total_duration:.2f}s")
                print(f"   - Costo: ${evaluation.cost_metrics.cost_per_query:.4f}")
            
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    # 3. Mostra riassunto
    print("\n📈 RIASSUNTO VALUTAZIONI:")
    summary = flow.get_evaluation_summary()
    print(f"   • Valutazioni totali: {summary.get('total_evaluations', 0)}")
    print(f"   • Score medio: {summary.get('averages', {}).get('overall_score', 0):.3f}")
    print(f"   • Migliore: {summary.get('best_performing', {}).get('score', 0):.3f}")
    print(f"   • Peggiore: {summary.get('worst_performing', {}).get('score', 0):.3f}")
    
    # 4. Dashboard
    print(f"\n📊 CREAZIONE DASHBOARD...")
    dashboard = EvaluationDashboard()
    dashboard.print_summary_console(days=1)
    
    # 5. Esporta report
    report_path = dashboard.export_detailed_report('json')
    print(f"\n💾 Report salvato in: {report_path}")

def demo_quick_evaluation():
    """
    Demo valutazione rapida senza flow completo
    """
    print("\n🔧 DEMO VALUTAZIONE RAPIDA")
    print("=" * 40)
    
    def mock_rag_method(query, subject, topic, **kwargs):
        """Mock del tuo metodo RAG"""
        return {
            "answer": f"Risposta RAG per '{query}' su {subject}/{topic}",
            "source_type": "RAG",
            "confidence": 0.9
        }
    
    # Valutazione rapida
    try:
        result = quick_evaluate(
            query="Cos'è il machine learning?",
            subject="AI",
            topic="machine_learning", 
            crew_method=mock_rag_method
        )
        
        print(f"✅ Valutazione completata!")
        print(f"   • Score: {result['score']:.3f}")
        print(f"   • Grado: {result['grade']}")
        print(f"   • Risultato: {result['result']['answer']}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    # Esegui demo completa
    demo_evaluation_system()
    
    # Esegui demo rapida
    demo_quick_evaluation()
    
    print(f"\n🎉 Demo completata! Ora puoi integrare il sistema nel tuo WebRAG_flow.py")
    print(f"\n📝 Per integrare nel tuo progetto:")
    print(f"   1. Copia la cartella 'evaluation' nel tuo src/")
    print(f"   2. Aggiungi 'from evaluation import auto_evaluate, FlowEvaluationMixin'")
    print(f"   3. Usa @auto_evaluate sui tuoi metodi crew")
    print(f"   4. Eredita da FlowEvaluationMixin nella tua classe Flow")
    print(f"   5. Usa dashboard.print_summary_console() per vedere risultati")
