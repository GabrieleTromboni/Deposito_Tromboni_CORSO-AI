"""
Test semplificato per RAGAS evaluation su Qdrant e FAISS
========================================================

Questo test semplificato dimostra l'utilizzo di RAGAS con entrambi i database,
risolvendo i problemi di compatibilità e creando un esempio funzionante.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# LangChain components
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Qdrant components
from qdrant_client import QdrantClient

# RAGAS evaluation
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

load_dotenv()

def test_qdrant_ragas():
    """Test RAGAS evaluation with Qdrant database"""
    print("🔍 Testing RAGAS with Qdrant Database")
    print("=" * 50)
    
    try:
        # Initialize components
        embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )
        
        llm = AzureChatOpenAI(
            model=os.getenv("CHAT_MODEL", "gpt-4o"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            temperature=0.1
        )
        
        # Connect to Qdrant
        qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        collection_name = "qdrant_database"  # Collezione già creata
        
        print("✅ Components initialized")
        
        # Test retrieval function
        def retrieve_from_qdrant(query: str, k: int = 3):
            """Retrieve documents from Qdrant"""
            try:
                # Get query embedding
                query_vector = embeddings.embed_query(query)
                
                # Search in Qdrant
                search_results = qdrant_client.search(
                    collection_name=collection_name,
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
        
        # Test retrieval
        print("\n🔍 Testing Qdrant retrieval...")
        test_query = "artificial intelligence"
        docs = retrieve_from_qdrant(test_query)
        
        if docs:
            print(f"✅ Retrieved {len(docs)} documents from Qdrant")
            print(f"📄 First document preview: {docs[0].page_content[:100]}...")
        else:
            print("❌ No documents retrieved from Qdrant")
            return False
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rispondi basandoti solo sul contesto fornito."),
            ("human", "Contesto:\n{context}\n\nDomanda: {question}")
        ])
        
        def rag_chain(question):
            docs = retrieve_from_qdrant(question)
            context = format_docs(docs)
            response = llm.invoke(prompt.format_messages(context=context, question=question))
            return response.content
        
        # Generate evaluation dataset
        print("\n📊 Generating evaluation dataset...")
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of AI?"
        ]
        
        dataset = []
        for question in questions:
            contexts = [doc.page_content for doc in retrieve_from_qdrant(question)]
            answer = rag_chain(question)
            
            entry = {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
            }
            dataset.append(entry)
        
        print(f"✅ Generated dataset with {len(dataset)} entries")
        
        # Run RAGAS evaluation (only metrics that don't require ground truth)
        print("\n🔍 Running RAGAS evaluation...")
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        metrics = [faithfulness, answer_relevancy]
        
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        
        # Display results
        df = result.to_pandas()
        print("\n📊 QDRANT RAGAS RESULTS")
        print("=" * 30)
        print(f"Faithfulness: {df['faithfulness'].mean():.3f}")
        print(f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}")
        
        # Save results
        df.to_csv("qdrant_ragas_results.csv", index=False)
        print("💾 Results saved to qdrant_ragas_results.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant RAGAS test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_ragas():
    """Test RAGAS evaluation with FAISS database"""
    print("\n🔍 Testing RAGAS with FAISS Database")
    print("=" * 50)
    
    try:
        # Initialize components
        embeddings = AzureOpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=os.getenv("AZURE_API_KEY"),
            openai_api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE")
        )
        
        llm = AzureChatOpenAI(
            model=os.getenv("CHAT_MODEL", "gpt-4o"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            temperature=0.1
        )
        
        # Load FAISS database
        persist_dir = "qdrant_database"  # Directory dove è salvato FAISS
        vector_store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✅ FAISS database loaded")
        
        # Test retrieval
        print("\n🔍 Testing FAISS retrieval...")
        test_query = "artificial intelligence"
        docs = vector_store.similarity_search(test_query, k=3)
        
        if docs:
            print(f"✅ Retrieved {len(docs)} documents from FAISS")
            print(f"📄 First document preview: {docs[0].page_content[:100]}...")
        else:
            print("❌ No documents retrieved from FAISS")
            return False
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rispondi basandoti solo sul contesto fornito."),
            ("human", "Contesto:\n{context}\n\nDomanda: {question}")
        ])
        
        def rag_chain(question):
            docs = vector_store.similarity_search(question, k=3)
            context = format_docs(docs)
            response = llm.invoke(prompt.format_messages(context=context, question=question))
            return response.content
        
        # Generate evaluation dataset
        print("\n📊 Generating evaluation dataset...")
        questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What are the applications of AI?"
        ]
        
        dataset = []
        for question in questions:
            contexts = [doc.page_content for doc in vector_store.similarity_search(question)]
            answer = rag_chain(question)
            
            entry = {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
            }
            dataset.append(entry)
        
        print(f"✅ Generated dataset with {len(dataset)} entries")
        
        # Run RAGAS evaluation
        print("\n🔍 Running RAGAS evaluation...")
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        
        metrics = [faithfulness, answer_relevancy]
        
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        
        # Display results
        df = result.to_pandas()
        print("\n📊 FAISS RAGAS RESULTS")
        print("=" * 30)
        print(f"Faithfulness: {df['faithfulness'].mean():.3f}")
        print(f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}")
        
        # Save results
        df.to_csv("faiss_ragas_results.csv", index=False)
        print("💾 Results saved to faiss_ragas_results.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ FAISS RAGAS test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_results():
    """Compare RAGAS results between Qdrant and FAISS"""
    print("\n🆚 Comparing RAGAS Results")
    print("=" * 40)
    
    try:
        if Path("qdrant_ragas_results.csv").exists() and Path("faiss_ragas_results.csv").exists():
            qdrant_df = pd.read_csv("qdrant_ragas_results.csv")
            faiss_df = pd.read_csv("faiss_ragas_results.csv")
            
            print("📊 COMPARISON SUMMARY")
            print("-" * 20)
            print(f"{'Metric':<20} {'Qdrant':<10} {'FAISS':<10} {'Winner'}")
            print("-" * 50)
            
            for metric in ['faithfulness', 'answer_relevancy']:
                if metric in qdrant_df.columns and metric in faiss_df.columns:
                    qdrant_score = qdrant_df[metric].mean()
                    faiss_score = faiss_df[metric].mean()
                    winner = "Qdrant" if qdrant_score > faiss_score else "FAISS"
                    
                    print(f"{metric:<20} {qdrant_score:<10.3f} {faiss_score:<10.3f} {winner}")
            
            return True
        else:
            print("❌ Results files not found for comparison")
            return False
            
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False

def main():
    """Run simplified RAGAS evaluation tests"""
    print("🚀 RAGAS Evaluation - Qdrant vs FAISS")
    print("=" * 60)
    
    # Test results
    results = []
    
    # Test Qdrant
    print("\n1️⃣ Testing Qdrant...")
    qdrant_success = test_qdrant_ragas()
    results.append(("Qdrant RAGAS", qdrant_success))
    
    # Test FAISS
    print("\n2️⃣ Testing FAISS...")
    faiss_success = test_faiss_ragas()
    results.append(("FAISS RAGAS", faiss_success))
    
    # Compare results
    if qdrant_success and faiss_success:
        print("\n3️⃣ Comparing Results...")
        comparison_success = compare_results()
        results.append(("Comparison", comparison_success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All RAGAS evaluation tests passed!")
        print("✅ Sistema di valutazione RAGAS completamente funzionante per entrambi i database!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
