"""
RAGAS Evaluation System for Qdrant Database
==========================================

This module provides comprehensive RAGAS evaluation capabilities for both Qdrant and FAISS
vector databases, integrated with the DatabaseCrew architecture.

KEY FEATURES:
- Support for both Qdrant and FAISS vector databases
- Integration with DatabaseCrew for database creation and management
- Comprehensive RAGAS metrics evaluation
- Hybrid search evaluation for Qdrant
- Performance benchmarking and comparison
- Automated evaluation dataset generation

EVALUATION METRICS:
- context_precision: Precision of retrieved contexts
- context_recall: Coverage of relevant contexts
- faithfulness: Answer groundedness in context
- answer_relevancy: Answer relevance to question
- answer_correctness: Answer accuracy (when ground truth available)

USAGE:
1. Create or connect to vector database using DatabaseCrew
2. Generate evaluation dataset with questions and ground truth
3. Run RAGAS evaluation with specified metrics
4. Analyze results and performance comparisons
"""

from __future__ import annotations

import os
import asyncio
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime

# LangChain components
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Qdrant components
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# RAGAS evaluation
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
)

# CrewAI and local imports
import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from src.progetto_crew_flows.crews.database_crew.database_crew import DatabaseCrew
    from src.progetto_crew_flows.tools.rag_tool import (
        QdrantSetting,
        intelligent_rag_search,
        get_qdrant_client,
        EMBEDDINGS
    )
    print("✅ Successfully imported DatabaseCrew and tools")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("🔧 Will create fallback components...")
    
    # Fallback components for testing
    class DatabaseCrew:
        def create_database(self, **kwargs):
            return "Database created successfully"
    
    class QdrantSetting:
        def __init__(self):
            self.collection_name = "test_collection"
    
    def intelligent_rag_search(query, collection_name, top_k):
        return f"Retrieved documents for: {query}"
    
    def get_qdrant_client():
        return None
    
    EMBEDDINGS = None

from dotenv import load_dotenv

load_dotenv()

@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation"""
    database_type: str = "qdrant"  # "qdrant" or "faiss"
    collection_name: str = "evaluation_collection"
    chunk_size: int = 800
    chunk_overlap: int = 100
    
    # Retrieval settings
    top_k: int = 5
    use_hybrid_search: bool = True  # Only for Qdrant
    
    # RAGAS settings
    evaluation_metrics: List[str] = None
    include_ground_truth: bool = True
    
    # Output settings
    output_dir: str = "evaluation_results"
    save_detailed_results: bool = True

    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "context_precision",
                "context_recall", 
                "faithfulness",
                "answer_relevancy",
                "answer_correctness"
            ]

class RAGASEvaluator:
    """
    RAGAS Evaluation system supporting both Qdrant and FAISS databases
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.database_crew = DatabaseCrew()
        self.llm = self._initialize_llm()
        self.embeddings = EMBEDDINGS
        self.evaluation_results = {}
        
        # Database-specific components
        self.vector_store = None
        self.qdrant_client = None
        self.retriever = None
        
    def _initialize_llm(self) -> AzureChatOpenAI:
        """Initialize Azure OpenAI LLM"""
        return AzureChatOpenAI(
            model=os.getenv("CHAT_MODEL", "gpt-4o"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            temperature=0.1
        )
    
    def setup_database(self, documents_path: str, force_recreate: bool = False) -> bool:
        """
        Setup vector database using DatabaseCrew
        """
        try:
            print(f"📋 Setting up {self.config.database_type.upper()} database...")
            
            if self.config.database_type == "qdrant":
                result = self.database_crew.create_database(
                    subject="evaluation",
                    topic="ragas_test",
                    database_type="qdrant",
                    collection_name=self.config.collection_name
                )
                
                if "success" in str(result).lower() or "created" in str(result).lower():
                    try:
                        self.qdrant_client = get_qdrant_client()
                    except TypeError:
                        # Fallback: create client directly
                        from qdrant_client import QdrantClient
                        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
                    print("✅ Qdrant database setup completed")
                    return True
                    
            elif self.config.database_type == "faiss":
                result = self.database_crew.create_database(
                    subject="evaluation",
                    topic="ragas_test", 
                    database_type="faiss"
                )
                
                if "success" in str(result).lower() or "created" in str(result).lower():
                    # Load FAISS vector store
                    persist_dir = os.getenv("RAG_DB_DIR", "RAG_database")
                    if Path(persist_dir).exists():
                        self.vector_store = FAISS.load_local(
                            persist_dir,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        print("✅ FAISS database setup completed")
                        return True
                        
            print(f"❌ Failed to setup {self.config.database_type} database")
            return False
            
        except Exception as e:
            print(f"❌ Database setup error: {e}")
            return False
    
    def _qdrant_retrieve(self, query: str, k: int) -> List[Document]:
        """Retrieve documents from Qdrant using hybrid search"""
        try:
            if self.config.use_hybrid_search:
                # Use intelligent RAG search with hybrid approach
                results = intelligent_rag_search(
                    query=query,
                    collection_name=self.config.collection_name,
                    top_k=k
                )
                
                # Parse results to extract documents
                documents = []
                if "Retrieved documents" in results:
                    # Extract document content from results
                    lines = results.split('\n')
                    current_doc = None
                    current_content = []
                    
                    for line in lines:
                        if line.startswith("**Document"):
                            if current_doc and current_content:
                                documents.append(Document(
                                    page_content='\n'.join(current_content),
                                    metadata={"source": f"doc_{len(documents)}"}
                                ))
                            current_content = []
                        elif line.strip() and not line.startswith("**") and not line.startswith("Query:"):
                            current_content.append(line.strip())
                    
                    # Add last document
                    if current_content:
                        documents.append(Document(
                            page_content='\n'.join(current_content),
                            metadata={"source": f"doc_{len(documents)}"}
                        ))
                
                return documents[:k]
                
            else:
                # Use direct Qdrant client for semantic search
                if not self.qdrant_client:
                    self.qdrant_client = get_qdrant_client()
                
                # Get query embedding
                query_vector = self.embeddings.embed_query(query)
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_vector,
                    limit=k
                )
                
                documents = []
                for hit in search_results:
                    documents.append(Document(
                        page_content=hit.payload.get("text", ""),
                        metadata=hit.payload.get("metadata", {})
                    ))
                
                return documents
                
        except Exception as e:
            print(f"❌ Qdrant retrieval error: {e}")
            return []
    
    def _faiss_retrieve(self, query: str, k: int) -> List[Document]:
        """Retrieve documents from FAISS"""
        try:
            if not self.vector_store:
                print("❌ FAISS vector store not initialized")
                return []
            
            # Use similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
            
        except Exception as e:
            print(f"❌ FAISS retrieval error: {e}")
            return []
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve documents based on database type"""
        k = k or self.config.top_k
        
        if self.config.database_type == "qdrant":
            return self._qdrant_retrieve(query, k)
        elif self.config.database_type == "faiss":
            return self._faiss_retrieve(query, k)
        else:
            raise ValueError(f"Unsupported database type: {self.config.database_type}")
    
    def create_rag_chain(self):
        """Create RAG chain for answer generation"""
        system_prompt = (
            "Sei un assistente esperto che risponde basandosi sul contesto fornito. "
            "IMPORTANTE: Rispondi SOLO utilizzando le informazioni contenute nel contesto. "
            "Se la risposta non è presente nel contesto, dillo chiaramente. "
            "Cita sempre le fonti quando possibile nel formato [source: X]."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human",
             "Domanda: {question}\n\n"
             "Contesto:\n{context}\n\n"
             "Istruzioni:\n"
             "1) Rispondi solo con informazioni contenute nel contesto\n"
             "2) Cita le fonti pertinenti\n"
             "3) Se la risposta non è nel contesto, dillo chiaramente")
        ])
        
        def format_docs(docs: List[Document]) -> str:
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", f"doc_{i}")
                formatted.append(f"[source: {source}] {doc.page_content}")
            return "\n\n".join(formatted)
        
        # Create the chain
        chain = (
            {
                "context": lambda x: format_docs(self.retrieve_documents(x["question"])),
                "question": lambda x: x["question"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def generate_evaluation_dataset(
        self, 
        questions: List[str], 
        ground_truth: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """Generate evaluation dataset for RAGAS"""
        print("📊 Generating evaluation dataset...")
        
        chain = self.create_rag_chain()
        dataset = []
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            
            # Get contexts and answer
            contexts = self.retrieve_documents(question)
            context_texts = [doc.page_content for doc in contexts]
            
            # Generate answer
            try:
                answer = chain.invoke({"question": question})
            except Exception as e:
                print(f"❌ Error generating answer for question {i}: {e}")
                answer = "Errore nella generazione della risposta"
            
            # Create dataset entry
            entry = {
                "user_input": question,
                "retrieved_contexts": context_texts,
                "response": answer,
            }
            
            # Add ground truth if available
            if ground_truth and question in ground_truth:
                entry["reference"] = ground_truth[question]
            
            dataset.append(entry)
        
        print(f"✅ Generated dataset with {len(dataset)} entries")
        return dataset
    
    def run_evaluation(
        self, 
        dataset: List[Dict], 
        save_results: bool = True
    ) -> Dict:
        """Run RAGAS evaluation"""
        print("🔍 Running RAGAS evaluation...")
        
        try:
            # Create evaluation dataset
            evaluation_dataset = EvaluationDataset.from_list(dataset)
            
            # Select metrics
            available_metrics = {
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "answer_correctness": answer_correctness,
            }
            
            metrics = []
            for metric_name in self.config.evaluation_metrics:
                if metric_name in available_metrics:
                    metrics.append(available_metrics[metric_name])
                else:
                    print(f"⚠️ Unknown metric: {metric_name}")
            
            # Skip answer_correctness if no ground truth
            has_ground_truth = all("reference" in item for item in dataset)
            if not has_ground_truth and answer_correctness in metrics:
                metrics.remove(answer_correctness)
                print("⚠️ Skipping answer_correctness - no ground truth available")
            
            print(f"📏 Evaluating with metrics: {[getattr(m, '__name__', str(m)) for m in metrics]}")
            
            # Run evaluation
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            
            # Convert to pandas and save
            df = result.to_pandas()
            
            if save_results:
                self._save_results(df, dataset)
            
            # Calculate summary statistics
            summary = self._calculate_summary(df)
            
            print("✅ Evaluation completed successfully")
            return {
                "detailed_results": df,
                "summary": summary,
                "dataset_size": len(dataset),
                "database_type": self.config.database_type
            }
            
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _save_results(self, df: pd.DataFrame, dataset: List[Dict]):
        """Save evaluation results"""
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_type = self.config.database_type
        
        # Save detailed results
        csv_file = output_dir / f"ragas_evaluation_{db_type}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Detailed results saved to: {csv_file}")
        
        # Save dataset
        if self.config.save_detailed_results:
            import json
            dataset_file = output_dir / f"evaluation_dataset_{db_type}_{timestamp}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"💾 Dataset saved to: {dataset_file}")
    
    def _calculate_summary(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        summary = {
            "database_type": self.config.database_type,
            "total_questions": len(df),
            "metrics": {}
        }
        
        # Calculate mean scores for each metric
        for col in df.columns:
            if col not in ["user_input", "retrieved_contexts", "response", "reference"]:
                if pd.api.types.is_numeric_dtype(df[col]):
                    summary["metrics"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
        
        return summary
    
    def compare_databases(
        self, 
        questions: List[str], 
        ground_truth: Optional[Dict[str, str]] = None,
        documents_path: str = None
    ) -> Dict:
        """Compare RAGAS evaluation between Qdrant and FAISS"""
        print("🆚 Running database comparison...")
        
        results = {}
        
        for db_type in ["qdrant", "faiss"]:
            print(f"\n📊 Evaluating {db_type.upper()} database...")
            
            # Create new config for this database type
            config = EvaluationConfig(
                database_type=db_type,
                collection_name=f"comparison_{db_type}",
                evaluation_metrics=self.config.evaluation_metrics,
                include_ground_truth=self.config.include_ground_truth
            )
            
            evaluator = RAGASEvaluator(config)
            
            # Setup database
            if documents_path and evaluator.setup_database(documents_path):
                # Generate dataset and run evaluation
                dataset = evaluator.generate_evaluation_dataset(questions, ground_truth)
                evaluation_result = evaluator.run_evaluation(dataset, save_results=True)
                results[db_type] = evaluation_result
            else:
                print(f"❌ Failed to setup {db_type} database")
                results[db_type] = None
        
        # Generate comparison report
        if all(results.values()):
            comparison_report = self._generate_comparison_report(results)
            return comparison_report
        else:
            print("❌ Comparison failed - not all databases were evaluated successfully")
            return results
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report between databases"""
        print("📊 Generating comparison report...")
        
        comparison = {
            "comparison_summary": {},
            "detailed_results": results,
            "recommendations": []
        }
        
        # Compare metrics
        for metric in self.config.evaluation_metrics:
            if metric == "answer_correctness":
                continue  # Skip if no ground truth
                
            qdrant_score = results.get("qdrant", {}).get("summary", {}).get("metrics", {}).get(metric, {}).get("mean")
            faiss_score = results.get("faiss", {}).get("summary", {}).get("metrics", {}).get(metric, {}).get("mean")
            
            if qdrant_score is not None and faiss_score is not None:
                comparison["comparison_summary"][metric] = {
                    "qdrant": qdrant_score,
                    "faiss": faiss_score,
                    "difference": qdrant_score - faiss_score,
                    "winner": "qdrant" if qdrant_score > faiss_score else "faiss"
                }
        
        # Generate recommendations
        if comparison["comparison_summary"]:
            qdrant_wins = sum(1 for m in comparison["comparison_summary"].values() if m["winner"] == "qdrant")
            faiss_wins = sum(1 for m in comparison["comparison_summary"].values() if m["winner"] == "faiss")
            
            if qdrant_wins > faiss_wins:
                comparison["recommendations"].append("✅ Qdrant shows better overall performance")
                comparison["recommendations"].append("🔍 Consider using Qdrant for hybrid search capabilities")
            elif faiss_wins > qdrant_wins:
                comparison["recommendations"].append("✅ FAISS shows better overall performance")
                comparison["recommendations"].append("🚀 FAISS may be faster for simple similarity search")
            else:
                comparison["recommendations"].append("⚖️ Both databases show similar performance")
                comparison["recommendations"].append("🎯 Choose based on specific requirements and infrastructure")
        
        # Save comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        import json
        report_file = output_dir / f"database_comparison_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Comparison report saved to: {report_file}")
        return comparison


def create_sample_evaluation_questions() -> Tuple[List[str], Dict[str, str]]:
    """Create sample questions for evaluation"""
    questions = [
        "Cos'è LangChain e come funziona?",
        "Quali sono i vantaggi di FAISS per la ricerca vettoriale?",
        "Come funziona il Retrieval-Augmented Generation?",
        "Cosa sono gli embeddings e come vengono utilizzati?",
        "Quali sono le differenze tra ricerca semantica e ricerca per parole chiave?",
        "Come si implementa un sistema RAG?",
        "Cos'è Qdrant e quali sono i suoi vantaggi?",
        "Come funziona la ricerca ibrida?",
        "Quali metriche utilizzare per valutare un sistema RAG?",
        "Come ottimizzare le performance di un vector database?"
    ]
    
    ground_truth = {
        questions[0]: "LangChain è un framework per sviluppare applicazioni basate su Large Language Models, fornendo catene, agenti, template di prompt e integrazioni.",
        questions[1]: "FAISS offre ricerca efficiente per similarità e clustering di vettori densi, supportando ricerca esatta e approssimata.",
        questions[2]: "RAG combina recupero di informazioni e generazione, selezionando documenti rilevanti e generando risposte basate su di essi.",
        questions[3]: "Gli embeddings sono rappresentazioni vettoriali di testo utilizzate per ricerca semantica e clustering.",
        questions[4]: "La ricerca semantica comprende il significato, mentre quella per parole chiave si basa su corrispondenze esatte."
    }
    
    return questions, ground_truth


# Usage example and main execution
def main():
    """Main execution function for RAGAS evaluation"""
    print("🚀 Starting RAGAS Evaluation for Qdrant Database")
    print("=" * 60)
    
    # Configuration
    config = EvaluationConfig(
        database_type="qdrant",  # Change to "faiss" for FAISS evaluation
        collection_name="ragas_evaluation",
        top_k=5,
        use_hybrid_search=True,
        evaluation_metrics=[
            "context_precision",
            "context_recall",
            "faithfulness", 
            "answer_relevancy"
        ],
        save_detailed_results=True
    )
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(config)
    
    # Setup database (specify your documents path)
    documents_path = "qdrant_database/documents"  # Change this to your documents folder
    
    if not evaluator.setup_database(documents_path, force_recreate=False):
        print("❌ Failed to setup database. Exiting.")
        return
    
    # Create evaluation questions
    questions, ground_truth = create_sample_evaluation_questions()
    
    # Generate evaluation dataset
    dataset = evaluator.generate_evaluation_dataset(questions, ground_truth)
    
    # Run evaluation
    results = evaluator.run_evaluation(dataset)
    
    # Display results
    if results:
        print("\n📊 EVALUATION RESULTS")
        print("=" * 40)
        
        summary = results["summary"]
        print(f"Database: {summary['database_type'].upper()}")
        print(f"Questions evaluated: {summary['total_questions']}")
        print("\nMetric Scores:")
        
        for metric, scores in summary["metrics"].items():
            print(f"  {metric}: {scores['mean']:.3f} (±{scores['std']:.3f})")
        
        print("\n✅ Evaluation completed successfully!")
        print("📁 Detailed results saved in 'evaluation_results' folder")


def run_database_comparison():
    """Run comparison between Qdrant and FAISS databases"""
    print("🆚 Starting Database Comparison Evaluation")
    print("=" * 50)
    
    config = EvaluationConfig(
        evaluation_metrics=[
            "context_precision",
            "context_recall", 
            "faithfulness",
            "answer_relevancy"
        ]
    )
    
    evaluator = RAGASEvaluator(config)
    
    # Create evaluation questions
    questions, ground_truth = create_sample_evaluation_questions()
    
    # Run comparison
    documents_path = "qdrant_database/documents"  # Adjust path as needed
    comparison_results = evaluator.compare_databases(
        questions=questions,
        ground_truth=ground_truth,
        documents_path=documents_path
    )
    
    # Display comparison results
    if comparison_results and "comparison_summary" in comparison_results:
        print("\n🏆 COMPARISON RESULTS")
        print("=" * 30)
        
        for metric, scores in comparison_results["comparison_summary"].items():
            print(f"{metric}:")
            print(f"  Qdrant: {scores['qdrant']:.3f}")
            print(f"  FAISS:  {scores['faiss']:.3f}")
            print(f"  Winner: {scores['winner'].upper()}")
            print()
        
        print("📋 RECOMMENDATIONS:")
        for rec in comparison_results["recommendations"]:
            print(f"  {rec}")


if __name__ == "__main__":
    # Run single database evaluation
    # main()
    
    # Or run database comparison
    run_database_comparison()
