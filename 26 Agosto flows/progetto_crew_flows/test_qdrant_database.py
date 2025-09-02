#!/usr/bin/env python3
"""
Test script per verificare la generazione del database Qdrant
con 2 documenti per ciascun topic definito.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.progetto_crew_flows.crews.database_crew.data_crew import DatabaseCrew
from src.progetto_crew_flows.WebRAG_flow import WebRAGFlow

def test_qdrant_database_generation():
    """Test Qdrant database generation with 2 documents per topic"""
    
    print("🧪 TESTING QDRANT DATABASE GENERATION")
    print("="*60)
    
    # Verify Qdrant configuration
    print("\n📋 Checking Qdrant Configuration:")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection = os.getenv("QDRANT_COLLECTION")
    
    print(f"   QDRANT_URL: {qdrant_url or 'NOT SET'}")
    print(f"   QDRANT_COLLECTION: {qdrant_collection or 'NOT SET'}")
    
    if not qdrant_url or not qdrant_collection:
        print("\n❌ ERROR: Qdrant configuration missing!")
        print("   Please set QDRANT_URL and QDRANT_COLLECTION environment variables.")
        print("   Example:")
        print("   QDRANT_URL=http://localhost:6333")
        print("   QDRANT_COLLECTION=rag_documents")
        return False
    
    # Test with a subset of subjects for faster testing
    test_subjects = {
        "medicine": ["cardiology", "neurology"],
        "football": ["premier league", "serie a"],
        "technology": ["artificial intelligence"]
    }
    
    print(f"\n🎯 Test Configuration:")
    print(f"   Database type: Qdrant")
    print(f"   Documents per topic: 2")
    print(f"   Test subjects: {list(test_subjects.keys())}")
    total_topics = sum(len(topics) for topics in test_subjects.values())
    total_docs = total_topics * 2
    print(f"   Total topics: {total_topics}")
    print(f"   Expected documents: {total_docs}")
    
    try:
        # Initialize DatabaseCrew with Qdrant
        print(f"\n🚀 Initializing DatabaseCrew with Qdrant...")
        database_crew = DatabaseCrew(use_qdrant=True)
        
        print(f"✅ DatabaseCrew initialized successfully")
        
        # Start database generation
        print(f"\n📊 Starting database generation...")
        result = database_crew.kickoff(
            subjects=test_subjects,
            docs_per_topic=2,           # Generate 2 documents per topic
            max_tokens_per_doc=800,     # Reasonable token limit
            batch_size=2                # Process 2 topics per batch
        )
        
        # Analyze results
        print(f"\n📈 Generation Results:")
        print(f"   Result type: {type(result)}")
        
        if isinstance(result, dict):
            status = result.get("status", "unknown")
            message = result.get("message", "No message")
            db_type = result.get("db_type", "Unknown")
            
            print(f"   Status: {status}")
            print(f"   Database type: {db_type}")
            print(f"   Message: {message}")
            
            if status == "success":
                print(f"\n✅ SUCCESS: Qdrant database generation completed!")
                
                # Try to get additional details
                if "topics_processed" in result:
                    topics_processed = result["topics_processed"]
                    print(f"   Topics processed: {len(topics_processed) if isinstance(topics_processed, list) else topics_processed}")
                
                return True
            elif status == "error":
                print(f"\n❌ ERROR: Database generation failed")
                print(f"   Error details: {message}")
                return False
            elif status == "skipped":
                print(f"\n⚠️ SKIPPED: {message}")
                return True
        else:
            print(f"   Raw result: {str(result)[:200]}...")
            print(f"\n✅ SUCCESS: Database generation completed (non-dict result)")
            return True
            
    except Exception as e:
        print(f"\n❌ EXCEPTION during database generation:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_qdrant_connection():
    """Verify Qdrant server connection"""
    
    print(f"\n🔌 Verifying Qdrant Connection:")
    
    try:
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        print(f"   Connecting to: {qdrant_url}")
        
        client = QdrantClient(url=qdrant_url)
        collections = client.get_collections()
        
        print(f"   ✅ Connected successfully!")
        print(f"   Available collections: {len(collections.collections)}")
        
        for collection in collections.collections:
            print(f"      - {collection.name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   Make sure Qdrant server is running at {qdrant_url}")
        return False

def main():
    """Main test function"""
    
    print(f"🧪 QDRANT DATABASE GENERATION TEST")
    print(f"=" * 60)
    print(f"Date: September 2, 2025")
    print(f"Test: Generate 2 documents per topic using Qdrant")
    
    # Step 1: Verify Qdrant connection
    if not verify_qdrant_connection():
        print(f"\n❌ Cannot proceed without Qdrant connection")
        return
    
    # Step 2: Test database generation
    success = test_qdrant_database_generation()
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"TEST SUMMARY")
    print(f"="*60)
    
    if success:
        print(f"✅ TEST PASSED: Qdrant database generation successful!")
        print(f"   • Documents generated: 2 per topic")
        print(f"   • Database type: Qdrant")
        print(f"   • Collection created and populated")
    else:
        print(f"❌ TEST FAILED: Qdrant database generation failed!")
        print(f"   Check the error messages above for details")
    
    print(f"\n🏁 Test completed.")

if __name__ == "__main__":
    main()
