"""
Dashboard e reporting per visualizzazione risultati valutazione CrewAI
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Conditional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ pandas not available. DataFrame functionality disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️ matplotlib/seaborn not available. Plotting functionality disabled.")

from .metrics import EvaluationResult

class EvaluationDashboard:
    """
    Dashboard per visualizzazione e analisi dei risultati di valutazione
    """
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Setup plotting style only if available
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")
    
    def load_evaluation_history(self, days: int = 30) -> List[EvaluationResult]:
        """Carica cronologia valutazioni degli ultimi N giorni"""
        history_file = self.results_dir / "evaluation_history.jsonl"
        
        if not history_file.exists():
            return []
        
        evaluations = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Converti timestamp string a datetime se necessario
                    if isinstance(data['timestamp'], str):
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    # Filtra per data
                    if data['timestamp'] >= cutoff_date:
                        evaluations.append(EvaluationResult.from_dict(data))
        except Exception as e:
            print(f"⚠️ Errore caricamento cronologia: {e}")
            
        return evaluations
    
    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """Genera report riassuntivo"""
        evaluations = self.load_evaluation_history(days)
        
        if not evaluations:
            return {
                "message": f"Nessuna valutazione trovata negli ultimi {days} giorni",
                "period": f"Ultimi {days} giorni",
                "total_evaluations": 0
            }
        
        # Statistiche base
        total_evaluations = len(evaluations)
        subjects = list(set(e.subject for e in evaluations))
        topics = list(set(e.topic for e in evaluations))
        source_types = list(set(e.source_type for e in evaluations))
        
        # Metriche aggregate
        avg_quality = sum(e.quality_metrics.overall_quality for e in evaluations) / total_evaluations
        avg_performance = sum(e.performance_metrics.total_duration for e in evaluations) / total_evaluations
        avg_cost = sum(e.cost_metrics.cost_per_query for e in evaluations) / total_evaluations
        avg_accuracy = sum(e.accuracy_metrics.overall_accuracy for e in evaluations) / total_evaluations
        avg_overall = sum(e.overall_score for e in evaluations) / total_evaluations
        
        # Distribuzione gradi
        grades = [e.grade for e in evaluations]
        grade_counts = {grade: grades.count(grade) for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']}
        grade_distribution = {k: v for k, v in grade_counts.items() if v > 0}
        
        # Performance per source type
        source_performance = {}
        for source_type in source_types:
            source_evals = [e for e in evaluations if e.source_type == source_type]
            if source_evals:
                source_performance[source_type] = {
                    'count': len(source_evals),
                    'avg_score': sum(e.overall_score for e in source_evals) / len(source_evals),
                    'avg_duration': sum(e.performance_metrics.total_duration for e in source_evals) / len(source_evals),
                    'avg_cost': sum(e.cost_metrics.cost_per_query for e in source_evals) / len(source_evals)
                }
        
        # Top e bottom performers
        best_eval = max(evaluations, key=lambda x: x.overall_score)
        worst_eval = min(evaluations, key=lambda x: x.overall_score)
        
        # Trends (se abbastanza dati)
        trends = {}
        if len(evaluations) >= 5:
            # Ordina per timestamp
            sorted_evals = sorted(evaluations, key=lambda x: x.timestamp)
            
            # Calcola trend punteggio (primi 50% vs ultimi 50%)
            mid_point = len(sorted_evals) // 2
            early_avg = sum(e.overall_score for e in sorted_evals[:mid_point]) / mid_point
            recent_avg = sum(e.overall_score for e in sorted_evals[mid_point:]) / (len(sorted_evals) - mid_point)
            score_trend = "📈 Miglioramento" if recent_avg > early_avg else "📉 Peggioramento" if recent_avg < early_avg else "➡️ Stabile"
            
            trends['score_trend'] = {
                'description': score_trend,
                'early_avg': round(early_avg, 3),
                'recent_avg': round(recent_avg, 3),
                'change': round(recent_avg - early_avg, 3)
            }
        
        return {
            'period': f"Ultimi {days} giorni",
            'generated_at': datetime.now().isoformat(),
            'overview': {
                'total_evaluations': total_evaluations,
                'unique_subjects': len(subjects),
                'unique_topics': len(topics),
                'source_types': source_types
            },
            'average_metrics': {
                'quality_score': round(avg_quality, 3),
                'performance_duration': round(avg_performance, 2),
                'cost_per_query': round(avg_cost, 4),
                'accuracy_score': round(avg_accuracy, 3),
                'overall_score': round(avg_overall, 3)
            },
            'grade_distribution': grade_distribution,
            'source_performance': source_performance,
            'best_performer': {
                'query': best_eval.query,
                'score': round(best_eval.overall_score, 3),
                'grade': best_eval.grade,
                'subject': best_eval.subject,
                'topic': best_eval.topic
            },
            'worst_performer': {
                'query': worst_eval.query,
                'score': round(worst_eval.overall_score, 3),
                'grade': worst_eval.grade,
                'subject': worst_eval.subject,
                'topic': worst_eval.topic
            },
            'trends': trends
        }
    
    def create_performance_charts(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Crea grafici di performance e li salva"""
        if not PANDAS_AVAILABLE or not PLOTTING_AVAILABLE:
            return {"error": "pandas o matplotlib non disponibili. Impossibile creare grafici."}
        
        evaluations = self.load_evaluation_history(30)
        
        if len(evaluations) < 2:
            return {"error": "Servono almeno 2 valutazioni per creare grafici"}
        
        # Prepara dati
        df = pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'overall_score': e.overall_score,
                'quality_score': e.quality_metrics.overall_quality,
                'accuracy_score': e.accuracy_metrics.overall_accuracy,
                'duration': e.performance_metrics.total_duration,
                'cost': e.cost_metrics.cost_per_query,
                'source_type': e.source_type,
                'grade': e.grade,
                'subject': e.subject
            }
            for e in evaluations
        ])
        
        charts_created = {}
        
        # 1. Trend punteggi nel tempo
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(df['timestamp'], df['overall_score'], marker='o', label='Overall Score')
        plt.plot(df['timestamp'], df['quality_score'], marker='s', alpha=0.7, label='Quality')
        plt.plot(df['timestamp'], df['accuracy_score'], marker='^', alpha=0.7, label='Accuracy')
        plt.title('Trend Punteggi nel Tempo')
        plt.xlabel('Data')
        plt.ylabel('Punteggio')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Distribuzione gradi
        plt.subplot(2, 2, 2)
        grade_counts = df['grade'].value_counts()
        plt.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%')
        plt.title('Distribuzione Gradi')
        
        # 3. Performance per source type
        plt.subplot(2, 2, 3)
        source_scores = df.groupby('source_type')['overall_score'].mean().sort_values(ascending=True)
        source_scores.plot(kind='barh')
        plt.title('Punteggio Medio per Tipo Source')
        plt.xlabel('Punteggio Medio')
        
        # 4. Relazione costo vs performance
        plt.subplot(2, 2, 4)
        plt.scatter(df['cost'], df['overall_score'], alpha=0.6, c=df['duration'], cmap='viridis')
        plt.xlabel('Costo per Query (USD)')
        plt.ylabel('Punteggio Overall')
        plt.title('Costo vs Performance')
        cbar = plt.colorbar()
        cbar.set_label('Durata (secondi)')
        
        plt.tight_layout()
        
        # Salva
        if save_path:
            chart_path = save_path
        else:
            chart_path = self.results_dir / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts_created['performance_overview'] = str(chart_path)
        
        # 2. Grafico dettagli qualità
        plt.figure(figsize=(10, 6))
        
        quality_metrics = []
        for e in evaluations:
            quality_metrics.append({
                'timestamp': e.timestamp,
                'clarity': e.quality_metrics.clarity_score,
                'coherence': e.quality_metrics.coherence_score,
                'completeness': e.quality_metrics.completeness_score,
                'relevance': e.quality_metrics.relevance_score,
                'accuracy': e.quality_metrics.accuracy_score,
                'language': e.quality_metrics.language_quality
            })
        
        quality_df = pd.DataFrame(quality_metrics)
        
        for metric in ['clarity', 'coherence', 'completeness', 'relevance', 'accuracy', 'language']:
            plt.plot(quality_df['timestamp'], quality_df[metric], marker='o', alpha=0.7, label=metric.title())
        
        plt.title('Dettaglio Metriche Qualità nel Tempo')
        plt.xlabel('Data')
        plt.ylabel('Punteggio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        quality_chart_path = self.results_dir / f"quality_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(quality_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts_created['quality_details'] = str(quality_chart_path)
        
        return charts_created
    
    def export_detailed_report(self, format: str = 'json') -> str:
        """Esporta report dettagliato in vari formati"""
        evaluations = self.load_evaluation_history(30)
        
        if format.lower() == 'csv':
            # Converti a DataFrame e esporta CSV
            data = []
            for e in evaluations:
                row = {
                    'timestamp': e.timestamp,
                    'query': e.query,
                    'subject': e.subject,
                    'topic': e.topic,
                    'source_type': e.source_type,
                    'overall_score': e.overall_score,
                    'grade': e.grade,
                    'quality_overall': e.quality_metrics.overall_quality,
                    'quality_clarity': e.quality_metrics.clarity_score,
                    'quality_coherence': e.quality_metrics.coherence_score,
                    'quality_completeness': e.quality_metrics.completeness_score,
                    'quality_relevance': e.quality_metrics.relevance_score,
                    'quality_accuracy': e.quality_metrics.accuracy_score,
                    'quality_language': e.quality_metrics.language_quality,
                    'performance_duration': e.performance_metrics.total_duration,
                    'performance_memory': e.performance_metrics.memory_usage_mb,
                    'cost_input_tokens': e.cost_metrics.input_tokens,
                    'cost_output_tokens': e.cost_metrics.output_tokens,
                    'cost_total_usd': e.cost_metrics.cost_per_query,
                    'accuracy_factual': e.accuracy_metrics.factual_accuracy,
                    'accuracy_source': e.accuracy_metrics.source_reliability,
                    'accuracy_citation': e.accuracy_metrics.citation_accuracy,
                    'accuracy_alignment': e.accuracy_metrics.topic_alignment
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_path = self.results_dir / f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            return str(csv_path)
            
        else:  # JSON
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_evaluations': len(evaluations),
                'period': 'Last 30 days',
                'summary': self.generate_summary_report(30),
                'detailed_evaluations': [e.to_dict() for e in evaluations]
            }
            
            json_path = self.results_dir / f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            return str(json_path)
    
    def print_summary_console(self, days: int = 7):
        """Stampa riassunto formattato in console"""
        summary = self.generate_summary_report(days)
        
        print("=" * 80)
        print(f"📊 EVALUATION DASHBOARD - {summary['period']}")
        print("=" * 80)
        
        if summary.get('total_evaluations', 0) == 0:
            print(summary.get('message', 'Nessun dato disponibile'))
            return
        
        # Overview
        overview = summary['overview']
        print(f"\n📈 OVERVIEW:")
        print(f"   • Valutazioni totali: {overview['total_evaluations']}")
        print(f"   • Soggetti unici: {overview['unique_subjects']}")
        print(f"   • Topic unici: {overview['unique_topics']}")
        print(f"   • Tipi source: {', '.join(overview['source_types'])}")
        
        # Metriche medie
        metrics = summary['average_metrics']
        print(f"\n📊 METRICHE MEDIE:")
        print(f"   • Punteggio Overall: {metrics['overall_score']:.3f}")
        print(f"   • Qualità: {metrics['quality_score']:.3f}")
        print(f"   • Accuratezza: {metrics['accuracy_score']:.3f}")
        print(f"   • Durata media: {metrics['performance_duration']:.2f}s")
        print(f"   • Costo medio: ${metrics['cost_per_query']:.4f}")
        
        # Distribuzione gradi
        grades = summary['grade_distribution']
        print(f"\n🎯 DISTRIBUZIONE GRADI:")
        for grade, count in grades.items():
            print(f"   • {grade}: {count} valutazioni")
        
        # Performance per source
        if summary['source_performance']:
            print(f"\n🔍 PERFORMANCE PER SOURCE TYPE:")
            for source, perf in summary['source_performance'].items():
                print(f"   • {source}:")
                print(f"     - Count: {perf['count']}")
                print(f"     - Avg Score: {perf['avg_score']:.3f}")
                print(f"     - Avg Duration: {perf['avg_duration']:.2f}s")
                print(f"     - Avg Cost: ${perf['avg_cost']:.4f}")
        
        # Best e worst
        best = summary['best_performer']
        worst = summary['worst_performer']
        print(f"\n🏆 MIGLIORE PERFORMANCE:")
        print(f"   • Query: {best['query'][:50]}...")
        print(f"   • Score: {best['score']} ({best['grade']})")
        print(f"   • Subject/Topic: {best['subject']}/{best['topic']}")
        
        print(f"\n⚠️ PEGGIORE PERFORMANCE:")
        print(f"   • Query: {worst['query'][:50]}...")
        print(f"   • Score: {worst['score']} ({worst['grade']})")
        print(f"   • Subject/Topic: {worst['subject']}/{worst['topic']}")
        
        # Trends
        if summary.get('trends'):
            print(f"\n📈 TRENDS:")
            for trend_name, trend_data in summary['trends'].items():
                print(f"   • {trend_name}: {trend_data['description']}")
                print(f"     - Cambiamento: {trend_data['change']:+.3f}")
        
        print("=" * 80)
