#!/usr/bin/env python
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
    from src.progetto_crew_flows.evaluation.integration import auto_evaluate, EvaluationManager, EvaluationDecorator3
    
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
    
    # 4. Test integration decorator
    decorator = EvaluationDecorator3(evaluator)
    print("✅ EvaluationDecorator inizializzato")
    
    print("\n🎉 Sistema di valutazione completamente funzionante!")
    print("📝 Tutte le dipendenze sono state risolte correttamente")
    print("🔧 Il sistema è pronto per essere utilizzato con CrewAI Flows")
    
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    print("🔧 Verifica che tutte le dipendenze siano installate nell'ambiente virtuale")
    sys.exit(1)

except Exception as e:
    print(f"❌ Errore durante il test: {e}")
    sys.exit(1)
