#!/usr/bin/env python3

import sys
sys.path.append('src')
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
API_VERSION = os.getenv('AZURE_API_VERSION')
API_KEY = os.getenv('AZURE_API_KEY')
CLIENT_AZURE = os.getenv('AZURE_API_BASE')

embeddings = AzureOpenAIEmbeddings(
    model='text-embedding-ada-002',
    openai_api_key=API_KEY,
    openai_api_version=API_VERSION,
    azure_endpoint=CLIENT_AZURE
)

# Load the vector store
persist_dir = 'RAG_database'
if Path(persist_dir).exists():
    vector_store = FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    print('=== ANALISI CHUNKS NEL DATABASE ===')
    print(f'Totale documenti nel vector store: {vector_store.index.ntotal}')
    
    # Test for different topics
    topics_to_test = ['medicine', 'football', 'technology']
    
    for topic in topics_to_test:
        print(f'\n--- Chunks per "{topic}" ---')
        test_docs = vector_store.similarity_search(topic, k=3)
        
        for i, doc in enumerate(test_docs):
            metadata = doc.metadata
            content_preview = doc.page_content[:60] + '...' if len(doc.page_content) > 60 else doc.page_content
            print(f'  Chunk {i+1}:')
            print(f'    Topic: {metadata.get("topic", "N/A")}')
            print(f'    Subject: {metadata.get("subject", "N/A")}')
            print(f'    Source: {metadata.get("source", "N/A")}')
            print(f'    Content: {content_preview}')
            print()
else:
    print('Database non trovato!')
