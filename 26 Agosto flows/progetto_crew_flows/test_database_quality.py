#!/usr/bin/env python3

import os
import sys
import ssl
from pathlib import Path

# Aggiungi il path del progetto
sys.path.append(str(Path(__file__).parent / "src"))

# Configurazione SSL
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

from src.progetto_crew_flows.tools.rag_tool import EMBEDDINGS
from langchain_community.vectorstores import FAISS

def test_database_quality():
    print("=== TEST QUALITÀ DATABASE RAG ===")
    
    # Usa la variabile globale EMBEDDINGS già inizializzata
    embeddings = EMBEDDINGS
    
    try:
        # Carica il database FAISS
        vectorstore = FAISS.load_local("RAG_database", embeddings, allow_dangerous_deserialization=True)
        print("✅ Database caricato con successo")
        
        # Test queries per ogni topic
        test_queries = [
            ("cardiology heart disease", "medicine", "cardiology"),
            ("neurology brain disorders", "medicine", "neurology"),
            ("psychiatry mental health", "medicine", "psychiatry"),
            ("premier league football", "football", "premier league"),
            ("serie a italian football", "football", "serie a"),
            ("artificial intelligence AI", "technology", "artificial intelligence"),
            ("blockchain cryptocurrency", "technology", "blockchain")
        ]
        
        print("\n🔍 QUALITÀ DEI CHUNKS:")
        
        for query, expected_subject, expected_topic in test_queries:
            print(f"\n--- Query: '{query}' ---")
            docs = vectorstore.similarity_search(query, k=2)
            
            if not docs:
                print(f"❌ Nessun documento trovato per {expected_topic}")
                continue
            
            for i, doc in enumerate(docs):
                subject = doc.metadata.get('subject', 'N/A')
                topic = doc.metadata.get('topic', 'N/A')
                doc_type = doc.metadata.get('doc_type', 'N/A')
                content_length = len(doc.page_content)
                
                # Verifica se il contenuto è ricco (non solo titoli)
                lines = doc.page_content.split('\n')
                substantial_lines = [line for line in lines if len(line.strip()) > 50]
                quality_score = len(substantial_lines) / max(len(lines), 1)
                
                print(f"  📄 Doc {i+1}: {subject} -> {topic} ({doc_type})")
                print(f"     Length: {content_length} chars")
                print(f"     Quality Score: {quality_score:.2f} (0-1)")
                
                # Mostra preview del contenuto
                preview = doc.page_content[:300].replace('\n', ' ')
                print(f"     Preview: {preview}...")
                
                # Verifica corretta classificazione
                if subject == expected_subject and topic == expected_topic:
                    print(f"     ✅ Correctly classified")
                else:
                    print(f"     ⚠️  Misclassified: expected {expected_subject}->{expected_topic}")
        
        # Statistiche generali
        print("\n📊 STATISTICHE GENERALI:")
        all_docs = vectorstore.similarity_search("", k=100)
        print(f"   Total chunks: {len(all_docs)}")
        
        # Distribuzione per subject/topic
        subjects = {}
        topics = {}
        doc_types = {}
        
        for doc in all_docs:
            subject = doc.metadata.get('subject', 'unknown')
            topic = doc.metadata.get('topic', 'unknown')
            doc_type = doc.metadata.get('doc_type', 'unknown')
            
            subjects[subject] = subjects.get(subject, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\n   📈 Per Subject:")
        for subject, count in sorted(subjects.items()):
            print(f"      {subject}: {count} chunks")
        
        print(f"\n   🎯 Per Topic:")
        for topic, count in sorted(topics.items()):
            print(f"      {topic}: {count} chunks")
        
        print(f"\n   📝 Per Document Type:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"      {doc_type}: {count} chunks")
        
        # Analisi qualità contenuti
        print(f"\n   🔍 Analisi Qualità:")
        total_chars = sum(len(doc.page_content) for doc in all_docs)
        avg_chars = total_chars / len(all_docs) if all_docs else 0
        print(f"      Avg chunk size: {avg_chars:.0f} characters")
        
        # Conta chunks con contenuto sostanziale
        substantial_chunks = 0
        for doc in all_docs:
            lines = doc.page_content.split('\n')
            substantial_lines = [line for line in lines if len(line.strip()) > 50]
            if len(substantial_lines) >= 3:  # Almeno 3 righe sostanziali
                substantial_chunks += 1
        
        quality_percentage = (substantial_chunks / len(all_docs)) * 100 if all_docs else 0
        print(f"      Substantial chunks: {substantial_chunks}/{len(all_docs)} ({quality_percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_quality()
