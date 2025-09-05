"""
Strumenti di integrazione per valutazione automatica dei CrewAI Flows
"""

import functools
import time
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

from .evaluator import CrewEvaluator
from .metrics import EvaluationResult

class EvaluationDecorator:
    """
    Decorator per valutazione automatica dei metodi dei Flows
    """
    
    def __init__(self, evaluator: CrewEvaluator):
        self.evaluator = evaluator
    
    def evaluate_flow_method(self, method_name: str, subject: str = None, topic: str = None):
        """
        Decorator per valutare automaticamente un metodo di Flow
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Estrai parametri dalla chiamata
                flow_instance = args[0] if args else None
                query = kwargs.get('query', 'Unknown query')
                
                # Usa subject/topic dal decorator o prova a estrarli dalla chiamata
                eval_subject = subject or kwargs.get('subject', 'Unknown')
                eval_topic = topic or kwargs.get('topic', 'Unknown')
                
                # Inizia valutazione
                context = self.evaluator.start_evaluation(query, eval_subject, eval_topic)
                
                try:
                    # Esegui metodo originale
                    result = func(*args, **kwargs)
                    
                    # Determina tipo di source dal metodo
                    source_type = self._determine_source_type(method_name, result)
                    
                    # Termina valutazione
                    evaluation = self.evaluator.end_evaluation(context, result, source_type)
                    
                    # Aggiungi valutazione al risultato se possibile
                    if hasattr(result, '__dict__'):
                        result._evaluation = evaluation
                    
                    return result
                    
                except Exception as e:
                    context['errors'].append(str(e))
                    print(f"❌ Errore durante valutazione di {method_name}: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def _determine_source_type(self, method_name: str, result: Any) -> str:
        """Determina il tipo di source dal nome del metodo"""
        if 'rag' in method_name.lower():
            return 'RAG'
        elif 'web' in method_name.lower() or 'search' in method_name.lower():
            return 'Web Search'
        elif hasattr(result, 'source_type'):
            return result.source_type
        else:
            return f'Flow Method: {method_name}'

@contextmanager
def evaluate_crew_execution(evaluator: CrewEvaluator, query: str, subject: str, topic: str, source_type: str):
    """
    Context manager per valutazione di esecuzione crew
    """
    context = evaluator.start_evaluation(query, subject, topic)
    
    try:
        yield context
    except Exception as e:
        context['errors'].append(str(e))
        raise
    finally:
        # Termina valutazione anche in caso di errore
        # Il risultato sarà None ma la valutazione viene comunque salvata
        evaluator.end_evaluation(context, None, source_type)

class FlowEvaluationMixin:
    """
    Mixin per aggiungere capacità di valutazione ai Flow
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluator = CrewEvaluator()
        self.evaluation_history = []
    
    def evaluate_last_execution(self) -> Optional[EvaluationResult]:
        """Ritorna l'ultima valutazione eseguita"""
        if self.evaluator.evaluation_history:
            return self.evaluator.evaluation_history[-1]
        return None
    
    def get_evaluation_summary(self, last_n: int = 5) -> Dict[str, Any]:
        """Riassunto delle ultime valutazioni"""
        return self.evaluator.get_evaluation_summary(last_n)
    
    def set_evaluation_config(self, config: Dict[str, Any]):
        """Configura il valutatore"""
        self.evaluator.config.update(config)

def auto_evaluate(subject: str = None, topic: str = None, source_type: str = None):
    """
    Decorator semplificato per auto-valutazione
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Cerca di ottenere un evaluator dall'istanza
            instance = args[0] if args else None
            evaluator = None
            
            if hasattr(instance, 'evaluator'):
                evaluator = instance.evaluator
            else:
                # Crea evaluator temporaneo
                evaluator = CrewEvaluator()
            
            # Estrai parametri
            query = kwargs.get('query', 'Auto-evaluation')
            eval_subject = subject or kwargs.get('subject', 'Auto')
            eval_topic = topic or kwargs.get('topic', 'Evaluation')
            eval_source_type = source_type or func.__name__
            
            # Valuta
            context = evaluator.start_evaluation(query, eval_subject, eval_topic)
            
            try:
                result = func(*args, **kwargs)
                evaluation = evaluator.end_evaluation(context, result, eval_source_type)
                
                # Salva valutazione nell'istanza se possibile
                if hasattr(instance, 'evaluation_history'):
                    instance.evaluation_history.append(evaluation)
                
                return result
                
            except Exception as e:
                context['errors'].append(str(e))
                evaluator.end_evaluation(context, None, eval_source_type)
                raise
                
        return wrapper
    return decorator

class BatchEvaluator:
    """
    Valutatore per test in batch di più query
    """
    
    def __init__(self, evaluator: CrewEvaluator = None):
        self.evaluator = evaluator or CrewEvaluator()
        self.batch_results = []
    
    def evaluate_queries_batch(self, queries: list, flow_method: Callable, **common_kwargs) -> list:
        """
        Valuta un batch di query con lo stesso metodo
        """
        results = []
        
        for i, query_data in enumerate(queries, 1):
            print(f"\n🔄 Batch evaluation {i}/{len(queries)}")
            
            if isinstance(query_data, str):
                query_data = {'query': query_data}
            
            # Combina kwargs comuni con quelli specifici della query
            kwargs = {**common_kwargs, **query_data}
            
            try:
                result = flow_method(**kwargs)
                results.append({
                    'query_data': query_data,
                    'result': result,
                    'evaluation': self.evaluator.evaluation_history[-1] if self.evaluator.evaluation_history else None,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'query_data': query_data,
                    'result': None,
                    'evaluation': None,
                    'error': str(e),
                    'success': False
                })
                print(f"❌ Errore nella query {i}: {e}")
        
        self.batch_results.extend(results)
        return results
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Riassunto del batch di valutazioni"""
        if not self.batch_results:
            return {"message": "Nessun batch eseguito"}
        
        total = len(self.batch_results)
        successful = sum(1 for r in self.batch_results if r['success'])
        failed = total - successful
        
        # Calcola statistiche sui successi
        success_evaluations = [r['evaluation'] for r in self.batch_results if r['success'] and r['evaluation']]
        
        if success_evaluations:
            avg_score = sum(e.overall_score for e in success_evaluations) / len(success_evaluations)
            avg_duration = sum(e.performance_metrics.total_duration for e in success_evaluations) / len(success_evaluations)
            best_score = max(e.overall_score for e in success_evaluations)
            worst_score = min(e.overall_score for e in success_evaluations)
        else:
            avg_score = avg_duration = best_score = worst_score = 0
        
        return {
            'total_queries': total,
            'successful': successful,
            'failed': failed,
            'success_rate': round(successful / total * 100, 1),
            'statistics': {
                'average_score': round(avg_score, 3),
                'average_duration': round(avg_duration, 2),
                'best_score': round(best_score, 3),
                'worst_score': round(worst_score, 3)
            }
        }

def create_evaluation_wrapper(flow_class):
    """
    Factory per creare una versione del Flow con valutazione automatica
    """
    class EvaluatedFlow(flow_class, FlowEvaluationMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    return EvaluatedFlow
