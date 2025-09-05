"""
Demo semplificato SENZA LLM - Solo metriche performance e costi
Test completo del sistema di valutazione base
"""

import sys
from pathlib import Path
import time
import json

# Aggiungi path del modulo evaluation  
sys.path.append(str(Path(__file__).parent / "src" / "progetto_crew_flows"))

try:
    from evaluation import (
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
        # Configurazione senza LLM per test veloce
        config = {
            'enable_quality_eval': False,    # Disabilita LLM per test
            'enable_performance_eval': True,
            'enable_cost_eval': True,
            'enable_accuracy_eval': False,   # Disabilita LLM per test
            'save_results': True,
            'detailed_logging': True
        }
        self.evaluator = CrewEvaluator(config)
        self.evaluation_history = []
    
    @auto_evaluate(source_type="RAG Database")
    def use_RAG(self, query: str, subject: str, topic: str) -> dict:
        """Mock del metodo RAG con valutazione automatica"""
        
        # Simula elaborazione
        time.sleep(1.2)  # Simula tempo di processing
        
        # Simula risultato RAG con più contenuto
        result = {
            "source_type": "RAG Database",
            "answer": f"Risposta RAG dettagliata per '{query}' nel contesto di {subject}/{topic}. Questa è una risposta completa e approfondita che include spiegazioni teoriche, esempi pratici e best practices. Il contenuto è basato su documentazione tecnica accurata e aggiornata.",
            "title": f"Guida Completa: {topic} in {subject}",
            "introduction": f"Introduzione approfondita a {topic} nel contesto di {subject}. Questa guida fornisce informazioni complete e aggiornate per aiutarti a comprendere e implementare {topic} efficacemente.",
            "sections": [
                {
                    'title': f"Fondamenti di {topic}",
                    'description': f"Concetti base e principi fondamentali di {topic}. Include definizioni, storia e contesto teorico necessario per una comprensione completa."
                },
                {
                    'title': f"Implementazione Pratica",
                    'description': f"Come applicare {topic} in scenari reali. Include esempi concreti, best practices e case studies di successo."
                },
                {
                    'title': f"Strumenti e Risorse",
                    'description': f"Strumenti, librerie e risorse utili per lavorare con {topic}. Include raccomandazioni specifiche e configurazioni ottimali."
                }
            ],
            'conclusion': f"Riepilogo dei punti chiave e prossimi passi per approfondire {topic}. Include riferimenti per ulteriori approfondimenti.",
            'target_audience': "Sviluppatori e professionisti IT",
            'sources': ['Database interno RAG', 'Documentazione tecnica', 'Knowledge base'],
            'confidence': 0.92
        }
        
        return result
    
    @auto_evaluate(source_type="Web Search")
    def use_web_search(self, query: str, subject: str, topic: str) -> dict:
        """Mock del metodo Web Search con valutazione automatica"""
        
        # Simula search sul web (più lento)
        time.sleep(2.1)
        
        result = {
            "source_type": "Web Search",
            "answer": f"Informazioni aggiornate dal web per '{query}' su {subject}/{topic}. Risultati basati su ricerche real-time che includono le ultime novità e trend del settore.",
            "title": f"Guida Aggiornata: {topic} - {subject}",
            "introduction": f"Ultime informazioni e novità su {topic} nel campo {subject}, raccolte da fonti web autorevoli e aggiornate.",
            "sections": [
                {
                    'title': f"Novità e Trend su {topic}",
                    'description': f"Ultime novità, trend e sviluppi recenti nel campo di {topic}. Include analisi delle tendenze attuali."
                },
                {
                    'title': f"Risorse Online",
                    'description': f"Migliori risorse, tutorial e documentazione online per {topic}. Include link e riferimenti verificati."
                }
            ],
            'conclusion': f"Sintesi delle informazioni più rilevanti e aggiornate su {topic} disponibili online.",
            'target_audience': "Tutti i livelli di esperienza",
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
    print("🚀 DEMO SISTEMA DI VALUTAZIONE CREWAI (NO LLM)")
    print("=" * 60)
    print("ℹ️  Valutazione LLM disabilitata per test veloce")
    print("    Focus su: Performance + Costi + Punteggio base")
    
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
    
    results_comparison = []
    
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
                print(f"      - Performance: {rag_eval.performance_metrics.total_duration:.2f}s")
                print(f"      - Costo: ${rag_eval.cost_metrics.cost_per_query:.4f}")
                print(f"      - Token: {rag_eval.cost_metrics.total_tokens}")
                
                results_comparison.append({
                    'query': test_query['query'],
                    'method': 'RAG',
                    'score': rag_eval.overall_score,
                    'duration': rag_eval.performance_metrics.total_duration,
                    'cost': rag_eval.cost_metrics.cost_per_query,
                    'tokens': rag_eval.cost_metrics.total_tokens
                })
            
            # Test Web Search  
            print("🌐 Testing Web Search...")
            web_result = flow.use_web_search(**test_query)
            web_eval = flow.get_last_evaluation()
            
            if web_eval:
                print(f"   ✅ Web Score: {web_eval.overall_score:.3f} ({web_eval.grade})")
                print(f"      - Performance: {web_eval.performance_metrics.total_duration:.2f}s")
                print(f"      - Costo: ${web_eval.cost_metrics.cost_per_query:.4f}")
                print(f"      - Token: {web_eval.cost_metrics.total_tokens}")
                
                results_comparison.append({
                    'query': test_query['query'],
                    'method': 'Web Search',
                    'score': web_eval.overall_score,
                    'duration': web_eval.performance_metrics.total_duration,
                    'cost': web_eval.cost_metrics.cost_per_query,
                    'tokens': web_eval.cost_metrics.total_tokens
                })
            
        except Exception as e:
            print(f"❌ Errore: {e}")
    
    # Mostra statistiche aggregate
    print(f"\n📈 STATISTICHE AGGREGATE:")
    total_evals = len(flow.evaluator.evaluation_history)
    if total_evals > 0:
        avg_score = sum(e.overall_score for e in flow.evaluator.evaluation_history) / total_evals
        avg_duration = sum(e.performance_metrics.total_duration for e in flow.evaluator.evaluation_history) / total_evals
        avg_cost = sum(e.cost_metrics.cost_per_query for e in flow.evaluator.evaluation_history) / total_evals
        total_cost = sum(e.cost_metrics.cost_per_query for e in flow.evaluator.evaluation_history)
        
        rag_evals = [e for e in flow.evaluator.evaluation_history if 'RAG' in e.source_type]
        web_evals = [e for e in flow.evaluator.evaluation_history if 'Web' in e.source_type]
        
        print(f"   • Valutazioni totali: {total_evals}")
        print(f"   • Score medio: {avg_score:.3f}")
        print(f"   • Durata media: {avg_duration:.2f}s")
        print(f"   • Costo medio per query: ${avg_cost:.4f}")
        print(f"   • Costo totale sessione: ${total_cost:.4f}")
        
        if rag_evals and web_evals:
            rag_avg_score = sum(e.overall_score for e in rag_evals) / len(rag_evals)
            web_avg_score = sum(e.overall_score for e in web_evals) / len(web_evals)
            rag_avg_duration = sum(e.performance_metrics.total_duration for e in rag_evals) / len(rag_evals)
            web_avg_duration = sum(e.performance_metrics.total_duration for e in web_evals) / len(web_evals)
            
            print(f"\n🔬 CONFRONTO RAG vs WEB SEARCH:")
            print(f"   • RAG - Score: {rag_avg_score:.3f} | Durata: {rag_avg_duration:.2f}s")
            print(f"   • Web - Score: {web_avg_score:.3f} | Durata: {web_avg_duration:.2f}s")
            
            if rag_avg_score > web_avg_score:
                print(f"   • 🏆 RAG vince per qualità (+{rag_avg_score-web_avg_score:.3f})")
            else:
                print(f"   • 🏆 Web Search vince per qualità (+{web_avg_score-rag_avg_score:.3f})")
                
            if rag_avg_duration < web_avg_duration:
                print(f"   • ⚡ RAG più veloce (-{web_avg_duration-rag_avg_duration:.2f}s)")
            else:
                print(f"   • ⚡ Web Search più veloce (-{rag_avg_duration-web_avg_duration:.2f}s)")
    
    # Tabella confronto
    if results_comparison:
        print(f"\n📊 TABELLA CONFRONTO DETTAGLIATA:")
        print("-" * 80)
        print(f"{'Query':<35} {'Method':<10} {'Score':<8} {'Duration':<8} {'Cost':<8}")
        print("-" * 80)
        for result in results_comparison:
            query_short = result['query'][:32] + "..." if len(result['query']) > 32 else result['query']
            print(f"{query_short:<35} {result['method']:<10} {result['score']:<8.3f} {result['duration']:<8.2f} ${result['cost']:<7.4f}")

def demo_quick_evaluation():
    """Demo valutazione rapida"""
    print(f"\n🔧 DEMO VALUTAZIONE RAPIDA")
    print("=" * 40)
    
    def mock_crew_method(query, subject, topic, **kwargs):
        """Mock metodo crew per test rapido"""
        time.sleep(0.8)
        return {
            "answer": f"Risposta rapida per '{query}' su {subject}/{topic}. Contenuto di test per verificare il funzionamento del sistema di valutazione.",
            "source_type": "Quick Test",
            "confidence": 0.85
        }
    
    try:
        print("⏱️  Esecuzione valutazione rapida...")
        
        result = quick_evaluate(
            query="Cos'è il machine learning?",
            subject="AI",
            topic="machine_learning",
            crew_method=mock_crew_method
        )
        
        print(f"✅ Valutazione rapida completata!")
        print(f"   • Score: {result['score']:.3f}")
        print(f"   • Grado: {result['grade']}")
        print(f"   • Risultato: {result['result']['answer'][:60]}...")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

def demo_dashboard():
    """Demo dashboard e reporting"""
    print(f"\n📊 DEMO DASHBOARD E REPORTING")
    print("=" * 40)
    
    try:
        # Dashboard
        dashboard = EvaluationDashboard()
        
        # Controlla se ci sono risultati
        history_file = dashboard.results_dir / "evaluation_history.jsonl"
        if history_file.exists():
            print("✅ File cronologia trovato")
            
            summary = dashboard.generate_summary_report(days=1)
            
            if summary.get('overview', {}).get('total_evaluations', 0) > 0:
                print(f"\n📈 DASHBOARD SUMMARY:")
                print(f"   • Valutazioni: {summary['overview']['total_evaluations']}")
                print(f"   • Score medio: {summary['average_metrics']['overall_score']:.3f}")
                print(f"   • Durata media: {summary['average_metrics']['performance_duration']:.2f}s")
                print(f"   • Costo medio: ${summary['average_metrics']['cost_per_query']:.4f}")
                
                # Distribuzione gradi
                print(f"\n🎯 Distribuzione gradi:")
                for grade, count in summary['grade_distribution'].items():
                    print(f"      {grade}: {count} valutazioni")
                
                # Esporta report
                try:
                    report_path = dashboard.export_detailed_report('json')
                    print(f"\n💾 Report JSON esportato: {Path(report_path).name}")
                    
                    csv_path = dashboard.export_detailed_report('csv')
                    print(f"💾 Report CSV esportato: {Path(csv_path).name}")
                    
                except Exception as e:
                    print(f"⚠️ Errore export: {e}")
                
            else:
                print("   ℹ️ Nessuna valutazione recente trovata per il dashboard")
        else:
            print("   ℹ️ Nessun file cronologia trovato")
            
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
        print(f"\n💡 CARATTERISTICHE TESTATE:")
        print(f"   ✅ Valutazione automatica con @auto_evaluate")
        print(f"   ✅ Metriche di performance (durata, memoria)")
        print(f"   ✅ Calcolo costi basato su token")
        print(f"   ✅ Sistema di grading (A+ to F)")
        print(f"   ✅ Confronto metodi (RAG vs Web Search)")
        print(f"   ✅ Dashboard e reporting")
        print(f"   ✅ Export risultati (JSON/CSV)")
        
        print(f"\n📝 INTEGRAZIONE NEL TUO PROGETTO:")
        print(f"   1. Copia cartella 'evaluation' in src/progetto_crew_flows/")
        print(f"   2. Nel tuo WebRAG_flow.py:")
        print(f"      from evaluation import auto_evaluate, FlowEvaluationMixin")
        print(f"   3. Aggiungi decorator ai metodi:")
        print(f"      @auto_evaluate(source_type='RAG')")
        print(f"   4. Monitora risultati:")
        print(f"      print_evaluation_summary()")
        
        print(f"\n🔧 CONFIGURAZIONE LLM:")
        print(f"   • Per abilitare valutazione qualità LLM:")
        print(f"     config['enable_quality_eval'] = True")
        print(f"   • Richiede Azure OpenAI configurato")
        print(f"   • In questo demo era disabilitato per test veloce")
        
        print(f"\n📚 Documentazione: EVALUATION_README.md")
        
    except Exception as e:
        print(f"❌ Errore durante demo: {e}")
        import traceback
        traceback.print_exc()
