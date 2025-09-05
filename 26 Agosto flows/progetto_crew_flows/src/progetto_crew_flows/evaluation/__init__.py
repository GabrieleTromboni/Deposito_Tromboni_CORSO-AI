"""
Modulo di valutazione per CrewAI Flows
Fornisce strumenti completi per valutare qualità, performance, costi e accuratezza
"""

from .metrics import (
    QualityMetrics,
    PerformanceMetrics,
    CostMetrics,
    AccuracyMetrics,
    EvaluationResult,
    EvaluationType
)

from .evaluator import CrewEvaluator

from .integration import (
    EvaluationDecorator,
    evaluate_crew_execution,
    FlowEvaluationMixin,
    auto_evaluate,
    BatchEvaluator,
    create_evaluation_wrapper
)

from .dashboard import EvaluationDashboard

__all__ = [
    # Metrics
    'QualityMetrics',
    'PerformanceMetrics', 
    'CostMetrics',
    'AccuracyMetrics',
    'EvaluationResult',
    'EvaluationType',
    
    # Core evaluator
    'CrewEvaluator',
    
    # Integration tools
    'EvaluationDecorator',
    'evaluate_crew_execution',
    'FlowEvaluationMixin',
    'auto_evaluate',
    'BatchEvaluator',
    'create_evaluation_wrapper',
    
    # Dashboard and reporting
    'EvaluationDashboard'
]

__version__ = "1.0.0"
__author__ = "CrewAI Evaluation System"

# Convenience imports for quick usage
def quick_evaluate(query: str, subject: str, topic: str, crew_method, **kwargs):
    """
    Valutazione rapida di un metodo crew
    
    Usage:
        result = quick_evaluate(
            query="Come cucinare la pasta",
            subject="cooking", 
            topic="pasta",
            crew_method=my_flow.use_RAG,
            other_param="value"
        )
    """
    evaluator = CrewEvaluator()
    context = evaluator.start_evaluation(query, subject, topic)
    
    try:
        result = crew_method(query=query, subject=subject, topic=topic, **kwargs)
        source_type = getattr(result, 'source_type', 'Unknown')
        evaluation = evaluator.end_evaluation(context, result, source_type)
        
        return {
            'result': result,
            'evaluation': evaluation,
            'score': evaluation.overall_score,
            'grade': evaluation.grade
        }
    except Exception as e:
        context['errors'].append(str(e))
        evaluation = evaluator.end_evaluation(context, None, 'Error')
        raise

def create_dashboard(results_dir: str = "evaluation_results") -> EvaluationDashboard:
    """
    Crea dashboard di valutazione
    """
    return EvaluationDashboard(results_dir)

def print_evaluation_summary(days: int = 7, results_dir: str = "evaluation_results"):
    """
    Stampa riassunto valutazioni in console
    """
    dashboard = EvaluationDashboard(results_dir)
    dashboard.print_summary_console(days)
