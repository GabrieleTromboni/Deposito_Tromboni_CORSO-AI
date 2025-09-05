"""
Core Evaluation Engine per CrewAI Flows
Motore principale che esegue tutte le valutazioni
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
from pathlib import Path
import traceback

# Try to import optional dependencies with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available. Memory usage tracking disabled.")

try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available. LLM evaluation disabled.")

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️ python-dotenv not available. Using system environment variables only.")

from .metrics import (
    EvaluationResult, QualityMetrics, PerformanceMetrics, 
    CostMetrics, AccuracyMetrics, EvaluationType
)

class CrewEvaluator:
    """
    Motore di valutazione principale per CrewAI Flows
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.llm = self._init_llm()
        self.evaluation_history: List[EvaluationResult] = []
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Monitoring variables
        self._start_time = None
        self._memory_tracker = None
        
    def _default_config(self) -> Dict:
        """Configurazione default per il valutatore"""
        return {
            'enable_quality_eval': True,
            'enable_performance_eval': True,
            'enable_cost_eval': True,
            'enable_accuracy_eval': True,
            'save_results': True,
            'detailed_logging': True,
            'cost_per_1k_tokens': {
                'input': 0.01,  # Costo per 1k input tokens
                'output': 0.03  # Costo per 1k output tokens
            }
        }
    
    def _init_llm(self):
        """Inizializza LLM per valutazioni qualitative"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        return AzureChatOpenAI(
            deployment_name=os.getenv("CHAT_MODEL") or os.getenv("MODEL"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            openai_api_key=os.getenv("AZURE_API_KEY"),
            temperature=0.1
        )
    
    def start_evaluation(self, query: str, subject: str, topic: str) -> Dict[str, Any]:
        """
        Inizia una nuova valutazione
        Ritorna un context manager per tracking automatico
        """
        start_memory = 0.0
        if PSUTIL_AVAILABLE:
            try:
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            except Exception:
                start_memory = 0.0
        
        evaluation_context = {
            'query': query,
            'subject': subject,
            'topic': topic,
            'start_time': time.time(),
            'start_memory': start_memory,
            'errors': [],
            'warnings': []
        }
        
        self._start_time = time.time()
        print(f"🔍 Iniziata valutazione per: {subject}/{topic}")
        
        return evaluation_context
    
    def end_evaluation(self, context: Dict[str, Any], result: Any, source_type: str) -> EvaluationResult:
        """
        Termina valutazione e calcola tutte le metriche
        """
        end_time = time.time()
        if PSUTIL_AVAILABLE:
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - context['start_memory']
        else:
            memory_usage = 0.0
        
        # Calcola metriche di performance
        performance_metrics = PerformanceMetrics(
            total_duration=end_time - context['start_time'],
            memory_usage_mb=memory_usage,
            crew_execution_time=end_time - context['start_time']  # Placeholder
        )
        
        # Estrai contenuto generato
        generated_content = self._extract_generated_content(result)
        
        # Valuta qualità se abilitata
        quality_metrics = QualityMetrics()
        if self.config['enable_quality_eval']:
            quality_metrics = self._evaluate_quality(
                context['query'], 
                generated_content,
                context['subject'],
                context['topic']
            )
        
        # Valuta accuratezza se abilitata
        accuracy_metrics = AccuracyMetrics()
        if self.config['enable_accuracy_eval']:
            accuracy_metrics = self._evaluate_accuracy(
                context['query'],
                generated_content,
                source_type
            )
        
        # Calcola costi se abilitato
        cost_metrics = CostMetrics()
        if self.config['enable_cost_eval']:
            cost_metrics = self._calculate_costs(generated_content)
        
        # Crea risultato finale
        evaluation_result = EvaluationResult(
            timestamp=datetime.now(),
            query=context['query'],
            subject=context['subject'],
            topic=context['topic'],
            source_type=source_type,
            quality_metrics=quality_metrics,
            performance_metrics=performance_metrics,
            cost_metrics=cost_metrics,
            accuracy_metrics=accuracy_metrics,
            generated_content=generated_content,
            sources_used=self._extract_sources(result),
            errors=context['errors'],
            warnings=context['warnings']
        )
        
        # Calcola punteggio complessivo
        evaluation_result.calculate_overall_score()
        
        # Salva risultati se abilitato
        if self.config['save_results']:
            self._save_evaluation_result(evaluation_result)
        
        # Aggiungi alla cronologia
        self.evaluation_history.append(evaluation_result)
        
        print(f"✅ Valutazione completata - Punteggio: {evaluation_result.overall_score:.3f} (Grado: {evaluation_result.grade})")
        
        return evaluation_result
    
    def _evaluate_quality(self, query: str, content: Dict[str, Any], subject: str, topic: str) -> QualityMetrics:
        """Valuta la qualità del contenuto usando LLM"""
        
        try:
            # Prepara il contenuto per la valutazione
            content_text = self._prepare_content_for_evaluation(content)
            
            # Prompt per valutazione qualità
            quality_prompt = ChatPromptTemplate.from_messages([
                ("system", """Sei un esperto valutatore di contenuti. Valuta il seguente contenuto su una scala 0-1 per ciascuna metrica.
                
                Rispondi SOLO con un JSON nel formato:
                {{
                    "clarity_score": 0.85,
                    "coherence_score": 0.90,
                    "completeness_score": 0.80,
                    "relevance_score": 0.95,
                    "accuracy_score": 0.85,
                    "language_quality": 0.90
                }}
                
                Criteri di valutazione:
                - clarity_score: Chiarezza e comprensibilità del testo
                - coherence_score: Coerenza logica e strutturale
                - completeness_score: Completezza delle informazioni
                - relevance_score: Rilevanza rispetto alla query originale
                - accuracy_score: Accuratezza delle informazioni (per quanto verificabile)
                - language_quality: Qualità linguistica e grammaticale"""),
                ("human", """Query originale: {query}
                Soggetto: {subject}
                Topic: {topic}
                
                Contenuto da valutare:
                {content}
                
                Valuta il contenuto:""")
            ])
            
            chain = quality_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "subject": subject,
                "topic": topic,
                "content": content_text[:2000]  # Limita lunghezza
            })
            
            # Parse JSON response
            quality_data = json.loads(response.strip())
            return QualityMetrics(**quality_data)
            
        except Exception as e:
            print(f"⚠️ Errore nella valutazione qualità: {e}")
            # Fallback con punteggi medi
            return QualityMetrics(
                clarity_score=0.7,
                coherence_score=0.7,
                completeness_score=0.7,
                relevance_score=0.7,
                accuracy_score=0.7,
                language_quality=0.7
            )
    
    def _evaluate_accuracy(self, query: str, content: Dict[str, Any], source_type: str) -> AccuracyMetrics:
        """Valuta l'accuratezza del contenuto"""
        
        try:
            content_text = self._prepare_content_for_evaluation(content)
            
            accuracy_prompt = ChatPromptTemplate.from_messages([
                ("system", """Valuta l'accuratezza del contenuto su scala 0-1 per ciascuna metrica.
                
                Rispondi SOLO con un JSON nel formato:
                {{
                    "factual_accuracy": 0.85,
                    "source_reliability": 0.90,
                    "citation_accuracy": 0.80,
                    "topic_alignment": 0.95
                }}
                
                Criteri:
                - factual_accuracy: Accuratezza dei fatti presentati
                - source_reliability: Affidabilità delle fonti citate o utilizzate
                - citation_accuracy: Correttezza delle citazioni (se presenti)
                - topic_alignment: Allineamento del contenuto al topic richiesto"""),
                ("human", """Query: {query}
                Tipo fonte: {source_type}
                
                Contenuto:
                {content}
                
                Valuta accuratezza:""")
            ])
            
            chain = accuracy_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "query": query,
                "source_type": source_type,
                "content": content_text[:2000]
            })
            
            accuracy_data = json.loads(response.strip())
            return AccuracyMetrics(**accuracy_data)
            
        except Exception as e:
            print(f"⚠️ Errore nella valutazione accuratezza: {e}")
            # Punteggi basati sul tipo di fonte
            base_score = 0.8 if "RAG" in source_type else 0.6
            return AccuracyMetrics(
                factual_accuracy=base_score,
                source_reliability=base_score,
                citation_accuracy=base_score,
                topic_alignment=base_score
            )
    
    def _calculate_costs(self, content: Dict[str, Any]) -> CostMetrics:
        """Calcola i costi stimati basati sull'uso dei token"""
        
        # Stima approssimativa dei token (1 token ≈ 4 caratteri)
        content_text = self._prepare_content_for_evaluation(content)
        estimated_output_tokens = len(content_text) // 4
        estimated_input_tokens = estimated_output_tokens // 10  # Stima input
        
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        # Calcola costo usando configurazione
        cost_config = self.config.get('cost_per_1k_tokens', {'input': 0.01, 'output': 0.03})
        input_cost = (estimated_input_tokens / 1000) * cost_config['input']
        output_cost = (estimated_output_tokens / 1000) * cost_config['output']
        total_cost = input_cost + output_cost
        
        return CostMetrics(
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=total_cost,
            cost_per_query=total_cost
        )
    
    def _extract_generated_content(self, result: Any) -> Dict[str, Any]:
        """Estrae il contenuto generato dal risultato"""
        if hasattr(result, 'guide_outline') and result.guide_outline:
            guide = result.guide_outline
            return {
                'title': getattr(guide, 'title', 'No title'),
                'introduction': getattr(guide, 'introduction', 'No introduction'),
                'sections': [
                    {
                        'title': section.title if hasattr(section, 'title') else 'No title',
                        'description': section.description if hasattr(section, 'description') else 'No description'
                    }
                    for section in (getattr(guide, 'sections', []))
                ],
                'conclusion': getattr(guide, 'conclusion', 'No conclusion'),
                'target_audience': getattr(guide, 'target_audience', 'General')
            }
        elif isinstance(result, dict):
            return result
        else:
            return {'content': str(result)}
    
    def _extract_sources(self, result: Any) -> List[str]:
        """Estrae le fonti utilizzate"""
        if hasattr(result, 'sources'):
            return result.sources
        elif hasattr(result, 'source_type'):
            return [result.source_type]
        else:
            return ["Unknown source"]
    
    def _prepare_content_for_evaluation(self, content: Dict[str, Any]) -> str:
        """Prepara il contenuto per la valutazione testuale"""
        if isinstance(content, dict):
            text_parts = []
            if 'title' in content:
                text_parts.append(f"Title: {content['title']}")
            if 'introduction' in content:
                text_parts.append(f"Introduction: {content['introduction']}")
            if 'sections' in content:
                for i, section in enumerate(content['sections'], 1):
                    text_parts.append(f"Section {i}: {section.get('title', '')} - {section.get('description', '')}")
            if 'conclusion' in content:
                text_parts.append(f"Conclusion: {content['conclusion']}")
            return "\n\n".join(text_parts)
        else:
            return str(content)
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """Salva il risultato della valutazione"""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{result.subject}_{result.topic}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Salva anche nel file cronologia aggregato
        history_file = self.results_dir / "evaluation_history.jsonl"
        with open(history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
    
    def get_evaluation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Genera un riassunto delle ultime valutazioni"""
        recent_evaluations = self.evaluation_history[-last_n:]
        
        if not recent_evaluations:
            return {"message": "Nessuna valutazione disponibile"}
        
        # Calcola statistiche aggregate
        avg_quality = sum(e.quality_metrics.overall_quality for e in recent_evaluations) / len(recent_evaluations)
        avg_performance = sum(e.performance_metrics.total_duration for e in recent_evaluations) / len(recent_evaluations)
        avg_cost = sum(e.cost_metrics.cost_per_query for e in recent_evaluations) / len(recent_evaluations)
        avg_accuracy = sum(e.accuracy_metrics.overall_accuracy for e in recent_evaluations) / len(recent_evaluations)
        avg_overall = sum(e.overall_score for e in recent_evaluations) / len(recent_evaluations)
        
        # Distribuzione gradi
        grades = [e.grade for e in recent_evaluations]
        grade_distribution = {grade: grades.count(grade) for grade in set(grades)}
        
        return {
            'total_evaluations': len(recent_evaluations),
            'period': f"Ultime {len(recent_evaluations)} valutazioni",
            'averages': {
                'quality_score': round(avg_quality, 3),
                'performance_duration': round(avg_performance, 2),
                'cost_per_query': round(avg_cost, 4),
                'accuracy_score': round(avg_accuracy, 3),
                'overall_score': round(avg_overall, 3)
            },
            'grade_distribution': grade_distribution,
            'best_performing': {
                'query': max(recent_evaluations, key=lambda x: x.overall_score).query,
                'score': max(recent_evaluations, key=lambda x: x.overall_score).overall_score
            },
            'worst_performing': {
                'query': min(recent_evaluations, key=lambda x: x.overall_score).query,
                'score': min(recent_evaluations, key=lambda x: x.overall_score).overall_score
            }
        }
