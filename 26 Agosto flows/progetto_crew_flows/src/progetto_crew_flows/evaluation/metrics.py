"""
CrewAI Flow Evaluation System
Sistema completo di valutazione per CrewAI Flows con metriche multiple
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import traceback
from enum import Enum

# Conditional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas not available. DataFrame export disabled.")

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback BaseModel for compatibility
    class BaseModel:
        pass

# Enum per i tipi di valutazione
class EvaluationType(Enum):
    QUALITY = "quality"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COST_EFFICIENCY = "cost_efficiency"

# Modelli Pydantic per le metriche
class QualityMetrics(BaseModel):
    """Metriche di qualità del contenuto"""
    clarity_score: float = 0.0  # 0-1: Chiarezza del contenuto
    coherence_score: float = 0.0  # 0-1: Coerenza logica
    completeness_score: float = 0.0  # 0-1: Completezza informazioni
    relevance_score: float = 0.0  # 0-1: Rilevanza rispetto alla query
    accuracy_score: float = 0.0  # 0-1: Accuratezza delle informazioni
    language_quality: float = 0.0  # 0-1: Qualità linguistica
    
    @property
    def overall_quality(self) -> float:
        """Calcola punteggio qualità complessivo"""
        scores = [
            self.clarity_score, self.coherence_score, self.completeness_score,
            self.relevance_score, self.accuracy_score, self.language_quality
        ]
        return sum(scores) / len(scores)

class PerformanceMetrics(BaseModel):
    """Metriche di performance tecnica"""
    total_duration: float = 0.0  # Durata totale in secondi
    crew_execution_time: float = 0.0  # Tempo esecuzione crew
    rag_retrieval_time: float = 0.0  # Tempo recupero RAG
    llm_inference_time: float = 0.0  # Tempo inferenza LLM
    memory_usage_mb: float = 0.0  # Utilizzo memoria in MB
    token_usage: Dict[str, int] = field(default_factory=dict)  # Uso token
    
    @property
    def tokens_per_second(self) -> float:
        """Calcola token processati per secondo"""
        total_tokens = sum(self.token_usage.values())
        return total_tokens / self.total_duration if self.total_duration > 0 else 0

class CostMetrics(BaseModel):
    """Metriche di costo"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_per_query: float = 0.0
    
class AccuracyMetrics(BaseModel):
    """Metriche di accuratezza"""
    factual_accuracy: float = 0.0  # 0-1: Accuratezza fattuale
    source_reliability: float = 0.0  # 0-1: Affidabilità fonti
    citation_accuracy: float = 0.0  # 0-1: Accuratezza citazioni
    topic_alignment: float = 0.0  # 0-1: Allineamento al topic
    
    @property
    def overall_accuracy(self) -> float:
        """Calcola accuratezza complessiva"""
        return (self.factual_accuracy + self.source_reliability + 
                self.citation_accuracy + self.topic_alignment) / 4

@dataclass
class EvaluationResult:
    """Risultato completo di una valutazione"""
    timestamp: datetime
    query: str
    subject: str
    topic: str
    source_type: str  # RAG_FAISS, RAG_QDRANT, WEB_SEARCH
    
    # Metriche
    quality_metrics: QualityMetrics
    performance_metrics: PerformanceMetrics
    cost_metrics: CostMetrics
    accuracy_metrics: AccuracyMetrics
    
    # Dati aggiuntivi
    generated_content: Dict[str, Any]
    sources_used: List[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Punteggi complessivi
    overall_score: float = 0.0
    grade: str = "F"  # A, B, C, D, F
    
    def calculate_overall_score(self):
        """Calcola punteggio complessivo pesato"""
        # Pesi per diverse metriche
        weights = {
            'quality': 0.4,
            'performance': 0.2,
            'accuracy': 0.3,
            'cost_efficiency': 0.1
        }
        
        # Normalizza costo (meno è meglio, max 1 USD per query)
        cost_efficiency = max(0, 1 - (self.cost_metrics.cost_per_query / 1.0))
        
        # Normalizza performance (max 60 secondi considerato "normale")
        performance_score = max(0, 1 - (self.performance_metrics.total_duration / 60.0))
        
        self.overall_score = (
            self.quality_metrics.overall_quality * weights['quality'] +
            performance_score * weights['performance'] +
            self.accuracy_metrics.overall_accuracy * weights['accuracy'] +
            cost_efficiency * weights['cost_efficiency']
        )
        
        # Assegna grado
        if self.overall_score >= 0.9:
            self.grade = "A"
        elif self.overall_score >= 0.8:
            self.grade = "B"
        elif self.overall_score >= 0.7:
            self.grade = "C"
        elif self.overall_score >= 0.6:
            self.grade = "D"
        else:
            self.grade = "F"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte risultato in dizionario per serializzazione"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'query': self.query,
            'subject': self.subject,
            'topic': self.topic,
            'source_type': self.source_type,
            'quality_metrics': self.quality_metrics.model_dump(),
            'performance_metrics': self.performance_metrics.model_dump(),
            'cost_metrics': self.cost_metrics.model_dump(),
            'accuracy_metrics': self.accuracy_metrics.model_dump(),
            'generated_content': self.generated_content,
            'sources_used': self.sources_used,
            'errors': self.errors,
            'warnings': self.warnings,
            'overall_score': self.overall_score,
            'grade': self.grade
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Crea risultato da dizionario"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            query=data['query'],
            subject=data['subject'],
            topic=data['topic'],
            source_type=data['source_type'],
            quality_metrics=QualityMetrics(**data['quality_metrics']),
            performance_metrics=PerformanceMetrics(**data['performance_metrics']),
            cost_metrics=CostMetrics(**data['cost_metrics']),
            accuracy_metrics=AccuracyMetrics(**data['accuracy_metrics']),
            generated_content=data['generated_content'],
            sources_used=data['sources_used'],
            errors=data.get('errors', []),
            warnings=data.get('warnings', []),
            overall_score=data['overall_score'],
            grade=data['grade']
        )
