"""
Test RAGAS Evaluation for Qdrant Database
========================================

This script tests the RAGAS evaluation system with both Qdrant and FAISS databases.
It includes comprehensive testing for evaluation metrics, database comparison, and error handling.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from rag_evaluation_qdrant import RAGASEvaluator, EvaluationConfig, create_sample_evaluation_questions

load_dotenv()

def test_qdrant_evaluation():
    """Test RAGAS evaluation with Qdrant database"""
    print("🧪 Testing Qdrant RAGAS Evaluation")
    print("=" * 50)
    
    config = EvaluationConfig(
        database_type="qdrant",
        collection_name="test_ragas_qdrant",
        top_k=3,
        use_hybrid_search=True,
        evaluation_metrics=[
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy"
        ],
        save_detailed_results=True
    )
    
    evaluator = RAGASEvaluator(config)
    
    # Check if documents exist
    documents_path = "qdrant_database/documents"
    if not Path(documents_path).exists():
        print(f"⚠️ Documents path not found: {documents_path}")
        print("📁 Available paths:")
        for path in Path(".").glob("**/documents"):
            print(f"  - {path}")
        return False
    
    # Setup database
    if not evaluator.setup_database(documents_path, force_recreate=False):
        print("❌ Failed to setup Qdrant database")
        return False
    
    # Test basic retrieval
    print("\n🔍 Testing document retrieval...")
    test_query = "Cos'è LangChain?"
    docs = evaluator.retrieve_documents(test_query, k=3)
    
    if docs:
        print(f"✅ Retrieved {len(docs)} documents")
        print(f"📄 First document preview: {docs[0].page_content[:100]}...")
    else:
        print("❌ No documents retrieved")
        return False
    
    # Test RAG chain
    print("\n🔗 Testing RAG chain...")
    chain = evaluator.create_rag_chain()
    
    try:
        answer = chain.invoke({"question": test_query})
        print(f"✅ Generated answer: {answer[:150]}...")
    except Exception as e:
        print(f"❌ RAG chain error: {e}")
        return False
    
    # Run mini evaluation
    print("\n📊 Running mini evaluation...")
    questions = [
        "Cos'è LangChain?",
        "Come funziona RAG?",
        "Cosa sono gli embeddings?"
    ]
    
    ground_truth = {
        questions[0]: "LangChain è un framework per sviluppare applicazioni con Large Language Models",
        questions[1]: "RAG combina recupero di informazioni e generazione di testo",
        questions[2]: "Gli embeddings sono rappresentazioni vettoriali di testo"
    }
    
    try:
        dataset = evaluator.generate_evaluation_dataset(questions, ground_truth)
        print(f"✅ Generated dataset with {len(dataset)} entries")
        
        # Run evaluation
        results = evaluator.run_evaluation(dataset[:2], save_results=True)  # Test with first 2 questions
        
        if results and "summary" in results:
            print("✅ Evaluation completed successfully")
            print(f"📊 Metrics evaluated: {len(results['summary']['metrics'])}")
            return True
        else:
            print("❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_evaluation():
    """Test RAGAS evaluation with FAISS database"""
    print("\n🧪 Testing FAISS RAGAS Evaluation")
    print("=" * 50)
    
    config = EvaluationConfig(
        database_type="faiss",
        top_k=3,
        evaluation_metrics=[
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy"
        ],
        save_detailed_results=True
    )
    
    evaluator = RAGASEvaluator(config)
    
    # Check if FAISS database exists
    faiss_db_path = os.getenv("RAG_DB_DIR", "RAG_database")
    if not Path(faiss_db_path).exists():
        print(f"⚠️ FAISS database not found: {faiss_db_path}")
        print("🔧 Creating FAISS database first...")
        
        documents_path = "qdrant_database/documents"
        if not evaluator.setup_database(documents_path, force_recreate=True):
            print("❌ Failed to setup FAISS database")
            return False
    else:
        # Load existing FAISS database
        if not evaluator.setup_database("", force_recreate=False):
            print("❌ Failed to load FAISS database")
            return False
    
    # Test basic retrieval
    print("\n🔍 Testing FAISS document retrieval...")
    test_query = "Cos'è LangChain?"
    docs = evaluator.retrieve_documents(test_query, k=3)
    
    if docs:
        print(f"✅ Retrieved {len(docs)} documents")
        print(f"📄 First document preview: {docs[0].page_content[:100]}...")
    else:
        print("❌ No documents retrieved from FAISS")
        return False
    
    # Run mini evaluation
    print("\n📊 Running FAISS mini evaluation...")
    questions = ["Cos'è LangChain?", "Come funziona RAG?"]
    
    try:
        dataset = evaluator.generate_evaluation_dataset(questions, None)
        results = evaluator.run_evaluation(dataset, save_results=True)
        
        if results and "summary" in results:
            print("✅ FAISS evaluation completed successfully")
            return True
        else:
            print("❌ FAISS evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ FAISS evaluation error: {e}")
        return False

def test_database_comparison():
    """Test database comparison functionality"""
    print("\n🆚 Testing Database Comparison")
    print("=" * 40)
    
    config = EvaluationConfig(
        evaluation_metrics=[
            "context_precision",
            "faithfulness",
            "answer_relevancy"
        ]
    )
    
    evaluator = RAGASEvaluator(config)
    
    # Simple test questions
    questions = [
        "Cos'è LangChain?",
        "Come funziona RAG?"
    ]
    
    ground_truth = {
        questions[0]: "LangChain è un framework per applicazioni LLM",
        questions[1]: "RAG combina recupero e generazione"
    }
    
    try:
        documents_path = "qdrant_database/documents"
        if not Path(documents_path).exists():
            print(f"⚠️ Documents path not found: {documents_path}")
            return False
        
        comparison_results = evaluator.compare_databases(
            questions=questions,
            ground_truth=ground_truth,
            documents_path=documents_path
        )
        
        if comparison_results and "comparison_summary" in comparison_results:
            print("✅ Database comparison completed successfully")
            
            print("\n📊 Comparison Summary:")
            for metric, scores in comparison_results["comparison_summary"].items():
                print(f"  {metric}: Qdrant={scores['qdrant']:.3f}, FAISS={scores['faiss']:.3f}")
            
            return True
        else:
            print("❌ Database comparison failed")
            return False
            
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False

def test_evaluation_components():
    """Test individual evaluation components"""
    print("\n🔧 Testing Evaluation Components")
    print("=" * 40)
    
    # Test configuration
    config = EvaluationConfig()
    print(f"✅ Default config created: {config.database_type}")
    
    # Test sample questions generation
    questions, ground_truth = create_sample_evaluation_questions()
    print(f"✅ Sample questions created: {len(questions)} questions, {len(ground_truth)} ground truth answers")
    
    # Test evaluator initialization
    try:
        evaluator = RAGASEvaluator(config)
        print("✅ RAGASEvaluator initialized successfully")
        
        # Test LLM initialization
        if evaluator.llm:
            print("✅ LLM initialized successfully")
        else:
            print("❌ LLM initialization failed")
            
        # Test embeddings
        if evaluator.embeddings:
            print("✅ Embeddings initialized successfully")
        else:
            print("❌ Embeddings initialization failed")
            
        return True
        
    except Exception as e:
        print(f"❌ Component test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAGAS Evaluation Tests")
    print("=" * 60)
    
    # Track test results
    tests = []
    
    # Test 1: Components
    print("\n1️⃣ Testing Components...")
    tests.append(("Components", test_evaluation_components()))
    
    # Test 2: Qdrant evaluation
    print("\n2️⃣ Testing Qdrant Evaluation...")
    tests.append(("Qdrant Evaluation", test_qdrant_evaluation()))
    
    # Test 3: FAISS evaluation
    print("\n3️⃣ Testing FAISS Evaluation...")
    tests.append(("FAISS Evaluation", test_faiss_evaluation()))
    
    # Test 4: Database comparison (if both passed)
    if tests[-1][1] and tests[-2][1]:  # If both Qdrant and FAISS tests passed
        print("\n4️⃣ Testing Database Comparison...")
        tests.append(("Database Comparison", test_database_comparison()))
    else:
        print("\n4️⃣ Skipping Database Comparison (prerequisite tests failed)")
        tests.append(("Database Comparison", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RAGAS evaluation system is ready.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == len(tests)

if __name__ == "__main__":
    # Set up environment
    load_dotenv()
    
    # Run tests
    success = main()
    
    if success:
        print("\n🚀 Ready to run full RAGAS evaluation!")
        print("Usage examples:")
        print("  python rag_evaluation_qdrant.py  # Run single database evaluation")
        print("  # Or modify main() to run run_database_comparison()")
    
    sys.exit(0 if success else 1)
