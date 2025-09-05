# 📊 Sistema di Valutazione CrewAI Flows

Un sistema completo per valutare **qualità**, **performance**, **costi** e **accuratezza** dei tuoi CrewAI Flows.

## 🚀 Quick Start

### 1. Integrazione Rapida

```python
from evaluation import auto_evaluate, FlowEvaluationMixin

class MyFlow(FlowEvaluationMixin, Flow):
    @auto_evaluate(source_type="RAG")
    def use_RAG(self, query, subject, topic):
        # Il tuo codice esistente
        return result
```

### 2. Valutazione Manuale

```python
from evaluation import quick_evaluate

result = quick_evaluate(
    query="Come funziona l'AI?",
    subject="technology", 
    topic="artificial_intelligence",
    crew_method=my_flow.use_RAG
)

print(f"Score: {result['score']:.3f} (Grade: {result['grade']})")
```

### 3. Dashboard e Report

```python
from evaluation import EvaluationDashboard, print_evaluation_summary

# Stampa riassunto in console
print_evaluation_summary(days=7)

# Dashboard completa
dashboard = EvaluationDashboard()
dashboard.print_summary_console()
charts = dashboard.create_performance_charts()
report_path = dashboard.export_detailed_report('csv')
```

## 📋 Componenti del Sistema

### 🎯 Metriche Valutate

#### 1. **Quality Metrics** (Qualità del contenuto)
- **Clarity**: Chiarezza e comprensibilità
- **Coherence**: Coerenza logica e strutturale  
- **Completeness**: Completezza delle informazioni
- **Relevance**: Rilevanza rispetto alla query
- **Accuracy**: Accuratezza delle informazioni
- **Language Quality**: Qualità linguistica

#### 2. **Performance Metrics** (Performance tecnica)
- **Duration**: Tempo di esecuzione totale
- **Memory Usage**: Uso memoria durante esecuzione
- **Token Usage**: Numero di token utilizzati

#### 3. **Cost Metrics** (Analisi costi)
- **Input/Output Tokens**: Conteggio token input/output
- **Estimated Cost**: Costo stimato in USD
- **Cost per Query**: Costo per singola query

#### 4. **Accuracy Metrics** (Accuratezza contenuto)
- **Factual Accuracy**: Accuratezza fattuale
- **Source Reliability**: Affidabilità delle fonti
- **Citation Accuracy**: Correttezza citazioni
- **Topic Alignment**: Allineamento al topic richiesto

### 🔧 Strumenti di Integrazione

#### 1. **Auto-Evaluate Decorator**
```python
@auto_evaluate(source_type="RAG", subject="tech", topic="AI")
def my_crew_method(self, query, **kwargs):
    return result
```

#### 2. **FlowEvaluationMixin**
```python
class MyFlow(FlowEvaluationMixin, Flow):
    def __init__(self):
        super().__init__()
        self.set_evaluation_config({
            'enable_quality_eval': True,
            'enable_cost_eval': True
        })
```

#### 3. **Context Manager**
```python
with evaluate_crew_execution(evaluator, query, subject, topic, "RAG") as ctx:
    result = crew.kickoff()
    # Valutazione automatica al termine
```

#### 4. **Batch Evaluator**
```python
batch_evaluator = BatchEvaluator()
results = batch_evaluator.evaluate_queries_batch(
    queries=["Query 1", "Query 2", "Query 3"],
    flow_method=my_flow.use_RAG,
    subject="tech"
)
```

### 📊 Dashboard e Reporting

#### 1. **Console Summary**
```python
print_evaluation_summary(days=7)
```
Mostra:
- Overview valutazioni
- Metriche medie
- Distribuzione gradi (A+, A, B+, B, C+, C, D, F)
- Performance per source type
- Best/worst performers
- Trends nel tempo

#### 2. **Performance Charts**
```python
dashboard = EvaluationDashboard()
charts = dashboard.create_performance_charts()
```
Genera:
- Trend punteggi nel tempo
- Distribuzione gradi (pie chart)
- Performance per source type (bar chart)
- Relazione costo vs performance (scatter plot)
- Dettaglio metriche qualità

#### 3. **Export Reports**
```python
# CSV per analisi dati
csv_path = dashboard.export_detailed_report('csv')

# JSON per integrazione
json_path = dashboard.export_detailed_report('json')
```

## 🔧 Configurazione

### Default Configuration
```python
{
    'enable_quality_eval': True,      # Valutazione qualità via LLM
    'enable_performance_eval': True,  # Tracking performance
    'enable_cost_eval': True,         # Calcolo costi
    'enable_accuracy_eval': True,     # Valutazione accuratezza
    'save_results': True,            # Salvataggio automatico
    'detailed_logging': True,        # Log dettagliati
    'cost_per_1k_tokens': {
        'input': 0.01,               # Costo per 1k input tokens
        'output': 0.03               # Costo per 1k output tokens
    }
}
```

### Personalizzazione
```python
evaluator = CrewEvaluator({
    'enable_quality_eval': False,  # Disabilita valutazione qualità
    'cost_per_1k_tokens': {
        'input': 0.005,            # Prezzi personalizzati
        'output': 0.015
    }
})
```

## 📁 Struttura File

```
evaluation/
├── __init__.py           # Exports e funzioni convenience
├── metrics.py           # Classi Pydantic per metriche
├── evaluator.py         # Motore principale di valutazione
├── integration.py       # Decorator e strumenti integrazione
└── dashboard.py         # Dashboard e reporting

evaluation_results/      # Cartella risultati (auto-creata)
├── evaluation_history.jsonl           # Cronologia completa
├── evaluation_tech_AI_20231201.json   # Risultati singoli
├── performance_charts_20231201.png    # Grafici performance
└── detailed_report_20231201.csv       # Report esportati
```

## 🎓 Esempi Pratici

### Esempio 1: Integrazione WebRAG Flow
```python
from evaluation import auto_evaluate, FlowEvaluationMixin

class WebRAG_flow(FlowEvaluationMixin, Flow):
    
    @Flow.listen("route_to_crew", condition=lambda result: result == "use_RAG")
    @auto_evaluate(source_type="RAG Database")
    def use_RAG(self, query: str, subject: str, topic: str):
        result = self.database_crew.kickoff({
            "query": query,
            "subject": subject, 
            "topic": topic
        })
        return result
    
    @Flow.listen("route_to_crew", condition=lambda result: result == "use_web_search")
    @auto_evaluate(source_type="Web Search")
    def use_web_search(self, query: str, subject: str, topic: str):
        result = self.search_crew.kickoff({
            "query": query,
            "subject": subject,
            "topic": topic  
        })
        return result

# Uso
flow = WebRAG_flow()
result = flow.kickoff({"query": "Come funziona Docker?"})

# Visualizza ultima valutazione
evaluation = flow.evaluate_last_execution()
print(f"Score: {evaluation.overall_score:.3f} ({evaluation.grade})")
```

### Esempio 2: Test di Confronto RAG vs Web Search
```python
from evaluation import BatchEvaluator

# Test queries
test_queries = [
    {"query": "Spiegami il machine learning", "subject": "AI", "topic": "ML"},
    {"query": "Come usare Docker", "subject": "DevOps", "topic": "containerization"},
    {"query": "Cos'è React?", "subject": "frontend", "topic": "react"}
]

# Testa RAG
batch_evaluator = BatchEvaluator()
rag_results = batch_evaluator.evaluate_queries_batch(
    queries=test_queries,
    flow_method=flow.use_RAG
)

# Testa Web Search  
web_results = batch_evaluator.evaluate_queries_batch(
    queries=test_queries,
    flow_method=flow.use_web_search
)

# Confronta risultati
rag_summary = batch_evaluator.get_batch_summary()
print(f"RAG Average Score: {rag_summary['statistics']['average_score']:.3f}")
print(f"Web Average Score: {web_summary['statistics']['average_score']:.3f}")
```

### Esempio 3: Monitoraggio Continuo
```python
# Setup valutazione continua
flow = WebRAG_flow()

# Esegui per una settimana...
for day in range(7):
    daily_queries = get_daily_queries()  # Le tue queries
    for query_data in daily_queries:
        flow.kickoff(query_data)

# Analisi settimanale
dashboard = EvaluationDashboard()
dashboard.print_summary_console(days=7)

# Grafici performance
charts = dashboard.create_performance_charts()
print(f"Charts saved: {charts}")

# Export per analisi esterna
report_path = dashboard.export_detailed_report('csv')
print(f"Data exported to: {report_path}")
```

## 📈 Interpretazione Risultati

### Scala Punteggi
- **0.90 - 1.00**: A+ (Eccellente)
- **0.85 - 0.89**: A  (Ottimo)  
- **0.80 - 0.84**: B+ (Buono+)
- **0.75 - 0.79**: B  (Buono)
- **0.70 - 0.74**: C+ (Sufficiente+)
- **0.65 - 0.69**: C  (Sufficiente)
- **0.60 - 0.64**: D  (Insufficiente)
- **< 0.60**:      F  (Molto insufficiente)

### Benchmark Performance
- **Durata < 5s**: Veloce ⚡
- **Durata 5-15s**: Normale ⏱️
- **Durata > 15s**: Lento 🐌

### Benchmark Costi  
- **< $0.01 per query**: Economico 💚
- **$0.01-0.05**: Moderato 💛
- **> $0.05**: Costoso 💰

## 🔧 Troubleshooting

### Errori Comuni

1. **"ModuleNotFoundError: evaluation"**
   ```bash
   # Assicurati che la cartella evaluation sia in src/progetto_crew_flows/
   ls src/progetto_crew_flows/evaluation/
   ```

2. **"LLM evaluation failed"**
   ```python
   # Verifica configurazione Azure OpenAI
   evaluator.config['enable_quality_eval'] = False  # Disabilita temporaneamente
   ```

3. **"No evaluations found"**
   ```python
   # Controlla cartella risultati
   dashboard = EvaluationDashboard("custom_results_dir")
   ```

### Debug Mode
```python
evaluator = CrewEvaluator({
    'detailed_logging': True,
    'save_results': True
})
```

## 🚀 Best Practices

1. **Inizia semplice**: Usa `@auto_evaluate` sui metodi principali
2. **Monitora trends**: Esegui `print_evaluation_summary()` regolarmente  
3. **Confronta configurazioni**: Usa `BatchEvaluator` per A/B testing
4. **Analizza costi**: Monitora `cost_per_query` per ottimizzazione
5. **Esporta dati**: Usa CSV per analisi avanzate con pandas/Excel

## 📞 Support

Per domande o problemi:
1. Controlla gli esempi in `evaluation_demo.py`
2. Verifica configurazione in `evaluator.py`
3. Usa `detailed_logging=True` per debug

---

**🎉 Congratulazioni! Ora hai un sistema di valutazione completo per i tuoi CrewAI Flows!**
