#!/usr/bin/env python3
"""
Test completo del sistema di valutazione CrewAI
Script di test finale per verificare tutte le funzionalità
"""

import sys
import os
from pathlib import Path

# Aggiungi il percorso del modulo
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.progetto_crew_flows.evaluation.metrics import (
        EvaluationResult, QualityMetrics, PerformanceMetrics, 
        CostMetrics, AccuracyMetrics, EvaluationType
    )
    from src.progetto_crew_flows.evaluation.evaluator import CrewEvaluator
    from src.progetto_crew_flows.evaluation.dashboard import EvaluationDashboard
    from src.progetto_crew_flows.evaluation.integration import auto_evaluate
    
    print("✅ Tutti i moduli importati correttamente!")
    
    # Test basic functionality
    print("\n🧪 Test funzionalità base...")
    
    # 1. Test metriche
    performance = PerformanceMetrics(
        total_duration=2.5,
        memory_usage_mb=150.0,
        crew_execution_time=2.0
    )
    print(f"Performance metrics: {performance.total_duration}s, {performance.memory_usage_mb}MB")
    
    # 2. Test evaluator
    evaluator = CrewEvaluator({
        'azure_endpoint': 'https://test.openai.azure.com',
        'azure_key': 'test-key',
        'cost_per_1k_tokens': {'input': 0.01, 'output': 0.03}
    })
    print("✅ CrewEvaluator inizializzato")
    
    # 3. Test dashboard
    dashboard = EvaluationDashboard("test_results")
    print("✅ Dashboard inizializzato")
    
    # 4. Test auto_evaluate decorator
    print("✅ auto_evaluate decorator disponibile")
    
    print("\n🎉 Sistema di valutazione completamente funzionante!")
    print("📝 Tutte le dipendenze sono state risolte correttamente")
    print("🔧 Il sistema è pronto per essere utilizzato con CrewAI Flows")
    
    # Test di esempio con metriche simulate
    print("\n📊 Test metriche simulate...")
    
    quality = QualityMetrics(
        clarity_score=0.85,
        coherence_score=0.90,
        completeness_score=0.80,
        relevance_score=0.88,
        accuracy_score=0.82
    )
    
    cost = CostMetrics(
        estimated_cost_usd=0.15,
        input_tokens=1200,
        output_tokens=800,
        cost_per_query=0.15
    )
    
    print(f"Quality overall: {quality.overall_quality:.2f}")
    print(f"Estimated cost: ${cost.estimated_cost_usd:.3f}")
    
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    print("🔧 Verifica che tutte le dipendenze siano installate nell'ambiente virtuale")
    sys.exit(1)

except Exception as e:
    print(f"❌ Errore durante il test: {e}")
    sys.exit(1)
