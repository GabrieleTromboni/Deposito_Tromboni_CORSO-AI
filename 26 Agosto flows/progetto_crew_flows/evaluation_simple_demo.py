"""
Demo semplificato del sistema di valutazione - Senza dipendenze CrewAI Flow
Dimostra tutte le funzionalità del sistema di valutazione
"""

import sys
from pathlib import Path
import time
import json

# Aggiungi path del modulo evaluation  
sys.path.append(str(Path(__file__).parent / "src" / "progetto_crew_flows"))

try:
    from src.progetto_crew_flows.evaluation import (
        CrewEvaluator, 
        auto_evaluate, 
        quick_evaluate,
        EvaluationDashboard,
        print_evaluation_summary
    )
except ImportError as e:
    print(f"❌ Errore import: {e}")
    print("Verifica che la cartella evaluation sia in src/progetto_crew_flows/")
    sys.exit(1)

class MockWebRAGFlow:
    """
    Mock del WebRAG Flow per dimostrare il sistema di valutazione
    """
    
    def __init__(self):
        self.evaluator = CrewEvaluator()
        self.evaluation_history = []
    
    @auto_evaluate(source_type="RAG Database")
    def use_RAG(self, query: str, subject: str, topic: str) -> dict:
        """Mock del metodo RAG con valutazione automatica"""
        
        # Simula elaborazione
        time.sleep(1.5)  # Simula tempo di processing
        
        # Simula risultato RAG
        result = {
            "source_type": "RAG Database",
            "answer": f"Risposta RAG dettagliata per '{query}' nel contesto di {subject}/{topic}",
            "guide_outline": type('obj', (object,), {
                'title': f"Guida Completa: {topic} in {subject}",
                'introduction': f"Introduzione approfondita a {topic} nel contesto di {subject}. Questa guida fornisce informazioni complete e aggiornate.",
                'sections': [
                    type('section', (object,), {
                        'title': f"Fondamenti di {topic}",
                        'description': f"Concetti base e principi fondamentali di {topic}. Include definizioni, storia e contesto teorico."
                    })(),
                    type('section', (object,), {
                        'title': f"Implementazione Pratica",
                        'description': f"Come applicare {topic} in scenari reali. Include esempi, best practices e case studies."
                    })(),
                    type('section', (object,), {
                        'title': f"Strumenti e Risorse",
                        'description': f"Strumenti, librerie e risorse utili per lavorare con {topic}."
                    })()
                ],
                'conclusion': f"Riepilogo dei punti chiave e prossimi passi per approfondire {topic}",
                'target_audience': "Sviluppatori e professionisti IT"
            })(),
            'sources': ['Database interno RAG', 'Documentazione tecnica', 'Knowledge base'],
            'confidence': 0.92
        }
        
        return result
    
    @auto_evaluate(source_type="Web Search")
    def use_web_search(self, query: str, subject: str, topic: str) -> dict:
        """Mock del metodo Web Search con valutazione automatica"""
        
        # Simula search sul web
        time.sleep(2.0)
        
        result = {
            "source_type": "Web Search",
            "answer": f"Informazioni aggiornate dal web per '{query}' su {subject}/{topic}",
            "guide_outline": type('obj', (object,), {
                'title': f"Guida Aggiornata: {topic} - {subject}",
                'introduction': f"Ultime informazioni e novità su {topic} nel campo {subject}, raccolte da fonti web autorevoli.",
                'sections': [
                    type('section', (object,), {
                        'title': f"Novità e Trend su {topic}",
                        'description': f"Ultime novità, trend e sviluppi recenti nel campo di {topic}."
                    })(),
                    type('section', (object,), {
                        'title': f"Risorse Online",
                        'description': f"Migliori risorse, tutorial e documentazione online per {topic}."
                    })()
                ],
                'conclusion': f"Sintesi delle informazioni più rilevanti e aggiornate su {topic}",
                'target_audience': "Tutti i livelli di esperienza"
            })(),
            'sources': ['Wikipedia', 'GitHub', 'Stack Overflow', 'Medium articles'],
            'confidence': 0.78
        }
        
        return result
    
    def get_last_evaluation(self):
        """Ottieni ultima valutazione"""
        if self.evaluator.evaluation_history:
            return self.evaluator.evaluation_history[-1]
        return None

def demo_basic_evaluation():
    """Demo base del sistema di valutazione"""
    print("🚀 DEMO SISTEMA DI VALUTAZIONE CREWAI")
    print("=" * 60)
    
    flow = MockWebRAGFlow()
    
    # Test queries
    test_queries = [
        {
            "query": "Come implementare machine learning in Python?",
            "subject": "programming", 
            "topic": "machine_learning"
        },
        {
            "query": "Guida ai database NoSQL",
            "subject": "database",
            "topic": "nosql"
        },
        {
            "query": "Tutorial Docker per principianti",
            "subject": "devops", 
            "topic": "containerization"
        }
    ]
    
    print(f"\n📊 Esecuzione {len(test_queries)} query di test...")
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)} ---")
        print(f"Query: {test_query['query']}")
        
        try:
            # Test RAG
            print("🔍 Testing RAG...")
            rag_result = flow.use_RAG(**test_query)
            rag_eval = flow.get_last_evaluation()
            
            if rag_eval:
                print(f"   ✅ RAG Score: {rag_eval.overall_score:.3f} ({rag_eval.grade})")
                print(f"      - Qualità: {rag_eval.quality_metrics.overall_quality:.3f}")
                print(f"      - Accuratezza: {rag_eval.accuracy_metrics.overall_accuracy:.3f}")
                print(f"      - Durata: {rag_eval.performance_metrics.total_duration:.2f}s")
                print(f"      - Costo: ${rag_eval.cost_metrics.cost_per_query:.4f}")
            
            # Test Web Search  
            print("🌐 Testing Web Search...")
            web_result = flow.use_web_search(**test_query)
            web_eval = flow.get_last_evaluation()
            
            if web_eval:
                print(f"   ✅ Web Score: {web_eval.overall_score:.3f} ({web_eval.grade})")
                print(f"      - Qualità: {web_eval.quality_metrics.overall_quality:.3f}")
                print(f"      - Accuratezza: {web_eval.accuracy_metrics.overall_accuracy:.3f}")
                print(f"      - Durata: {web_eval.performance_metrics.total_duration:.2f}s")
                print(f"      - Costo: ${web_eval.cost_metrics.cost_per_query:.4f}")
            
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    # Mostra statistiche aggregate
    print(f"\n📈 STATISTICHE AGGREGATE:")
    total_evals = len(flow.evaluator.evaluation_history)
    if total_evals > 0:
        avg_score = sum(e.overall_score for e in flow.evaluator.evaluation_history) / total_evals
        avg_duration = sum(e.performance_metrics.total_duration for e in flow.evaluator.evaluation_history) / total_evals
        avg_cost = sum(e.cost_metrics.cost_per_query for e in flow.evaluator.evaluation_history) / total_evals
        
        rag_evals = [e for e in flow.evaluator.evaluation_history if 'RAG' in e.source_type]
        web_evals = [e for e in flow.evaluator.evaluation_history if 'Web' in e.source_type]
        
        print(f"   • Valutazioni totali: {total_evals}")
        print(f"   • Score medio: {avg_score:.3f}")
        print(f"   • Durata media: {avg_duration:.2f}s")
        print(f"   • Costo medio: ${avg_cost:.4f}")
        
        if rag_evals and web_evals:
            rag_avg = sum(e.overall_score for e in rag_evals) / len(rag_evals)
            web_avg = sum(e.overall_score for e in web_evals) / len(web_evals)
            print(f"   • RAG avg: {rag_avg:.3f} | Web avg: {web_avg:.3f}")
            
            better_method = "RAG" if rag_avg > web_avg else "Web Search"
            print(f"   • 🏆 Migliore metodo: {better_method}")

def demo_quick_evaluation():
    """Demo valutazione rapida"""
    print(f"\n🔧 DEMO VALUTAZIONE RAPIDA")
    print("=" * 40)
    
    def mock_crew_method(query, subject, topic, **kwargs):
        """Mock metodo crew per test rapido"""
        time.sleep(0.8)
        return {
            "answer": f"Risposta per '{query}' su {subject}/{topic}",
            "source_type": "Quick Test",
            "confidence": 0.85
        }
    
    try:
        result = quick_evaluate(
            query="Cos'è il machine learning?",
            subject="AI",
            topic="machine_learning",
            crew_method=mock_crew_method
        )
        
        print(f"✅ Valutazione rapida completata!")
        print(f"   • Score: {result['score']:.3f}")
        print(f"   • Grado: {result['grade']}")
        print(f"   • Risultato: {result['result']['answer']}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

def demo_dashboard():
    """Demo dashboard e reporting"""
    print(f"\n📊 DEMO DASHBOARD")
    print("=" * 30)
    
    try:
        # Stampa summary se ci sono valutazioni
        print("📋 Summary valutazioni:")
        print_evaluation_summary(days=1)
        
        # Dashboard dettagliata
        dashboard = EvaluationDashboard()
        summary = dashboard.generate_summary_report(days=1)
        
        if summary.get('overview', {}).get('total_evaluations', 0) > 0:
            print(f"\n📈 Report Dashboard:")
            print(f"   • Valutazioni: {summary['overview']['total_evaluations']}")
            print(f"   • Score medio: {summary['average_metrics']['overall_score']:.3f}")
            print(f"   • Migliore: {summary['best_performer']['score']:.3f}")
            print(f"   • Peggiore: {summary['worst_performer']['score']:.3f}")
            
            # Esporta report
            report_path = dashboard.export_detailed_report('json')
            print(f"   • Report salvato: {Path(report_path).name}")
        else:
            print("   ℹ️ Nessuna valutazione recente trovata")
            
    except Exception as e:
        print(f"❌ Errore dashboard: {e}")

if __name__ == "__main__":
    try:
        # Demo principale
        demo_basic_evaluation()
        
        # Demo valutazione rapida
        demo_quick_evaluation()
        
        # Demo dashboard
        demo_dashboard()
        
        print(f"\n🎉 DEMO COMPLETATO!")
        print("=" * 60)
        print(f"✅ Sistema di valutazione CrewAI funzionante!")
        print(f"\n📝 PROSSIMI PASSI:")
        print(f"   1. Integra nel tuo WebRAG_flow.py:")
        print(f"      - Aggiungi: from evaluation import auto_evaluate, FlowEvaluationMixin")
        print(f"      - Usa: @auto_evaluate sui tuoi metodi crew")
        print(f"   2. Monitora performance:")
        print(f"      - Esegui: print_evaluation_summary() periodicamente")
        print(f"   3. Analizza risultati:")
        print(f"      - Usa: dashboard.create_performance_charts()")
        print(f"      - Esporta: dashboard.export_detailed_report('csv')")
        print(f"\n📚 Documentazione completa: EVALUATION_README.md")
        
    except Exception as e:
        print(f"❌ Errore durante demo: {e}")
        import traceback
        traceback.print_exc()
