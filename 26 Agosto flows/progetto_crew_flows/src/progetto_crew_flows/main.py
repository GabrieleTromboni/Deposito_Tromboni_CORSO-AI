'''
Flow schema:

    Inputs: query from user about search topic, complex answer with topic inside.
    First step: scan query and extract the search topic from it.
    Second step: validate the search topic, is it into the allowed topic list? Topic list predefined.
    Third step: if valid, proceed with a RAG (Retrieval-Augmented Generation) process.
        The topic will be used to retrieve relevant documents and generate a response.
        RAG as med_crew.py (rielaborate prompt, search results into its vector database, generate answer as GuideOutline class).
    Fourth step: if not valid, start a search_crew to find relevant informations on the web about that topic (use search_crew.py)
    Fifth step: return the final response to the user and save it using OutputGuideline class independently if coming from RAG or search_crew.
    Final steps: kickoff() and plot()

Instructions for use:
    Generate documents into vector database using an LLM (Large Language Model) of different topics following what is done in rag_faiss_lmstudio.py
    and save the full topics list to know which topics are available in order to validate the search topic and retrieve relevant documents with RAG.

    First and second steps must be done with an LLM (Large Language Model) with care and precision.
    '''

from progetto_crew_flows.crews.database_crew.database_crew import DatabaseCrew, create_database_crew
from progetto_crew_flows.WebRAG_flow import WebRAGFlow 

def main():
    """Main execution function"""
    
    # Vector Database Selection
    print("\n" + "="*60)
    print("VECTOR DATABASE SETUP")
    print("="*60)

    print("\n🔧 Vector Database Selection:")
    print("Choose your preferred vector database:")
    print("1. FAISS (Local file-based storage)")
    print("2. Qdrant (Vector database server)")
    
    while True:
        choice = input("\nEnter your choice (1 for FAISS, 2 for Qdrant): ").strip()
        if choice == "1":
            database_type = "faiss"
            db_name = "FAISS"
            break
        elif choice == "2":
            database_type = "qdrant"
            db_name = "Qdrant"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print(f"✅ Selected: {db_name} vector database")
    
    # Database Operation Selection
    print("\n🔧 Operation Selection:")
    print("Choose what you want to do:")
    print("1. Create/Initialize Database")
    print("2. Query Existing Database (RAG)")
    print("3. Both (Create if needed, then query)")
    
    while True:
        op_choice = input("\nEnter your choice (1, 2, or 3): ").strip()
        if op_choice in ["1", "2", "3"]:
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Initialize DatabaseCrew
    database_crew = create_database_crew()
    
    # Handle database creation if requested
    if op_choice in ["1", "3"]:
        print(f"\n🔄 Creating {db_name} database...")
        
        # For demo, create databases for all subjects
        subjects_topics = {
            "medicine": ["cardiology", "neurology", "oncology"],
            "football": ["premier league", "serie a", "champions league"], 
            "technology": ["artificial intelligence", "machine learning", "blockchain"]
        }
        
        for subject, topics in subjects_topics.items():
            for topic in topics:
                print(f"   Creating database for {subject} - {topic}...")
                try:
                    result = database_crew.create_database(
                        subject=subject,
                        topic=topic,
                        database_type=database_type
                    )
                    if result.get('status') == 'success':
                        print(f"   ✅ Created {subject}/{topic}")
                    else:
                        print(f"   ⚠️ Issue with {subject}/{topic}: {result}")
                except Exception as e:
                    print(f"   ❌ Error creating {subject}/{topic}: {e}")
        
        print(f"✅ {db_name} database creation completed!")
    
    # Handle querying if requested  
    if op_choice in ["2", "3"]:
        print(f"\n🔍 Starting RAG Query System with {db_name}...")
        
        # Available databases for the flow
        available_databases = [database_type]
        
        print("\nAvailable subjects and topics:")
        subjects_topics = {
            "medicine": ["cardiology", "neurology", "oncology"],
            "football": ["premier league", "serie a", "champions league"],
            "technology": ["artificial intelligence", "machine learning", "blockchain"]
        }
        
        for subject, topics in subjects_topics.items():
            print(f"\n{subject.upper()}:")
            for topic in topics:
                print(f"  - {topic}")
        
        print("\n" + "-"*60)
        print("You can ask questions about any of the above subjects and topics.")
        print(f"RAG will use {db_name} database for retrieval.")
        print("Type 'exit' to quit\n")
        
        while True:
            # Get user input
            query = input("\nEnter your query (or 'exit' to quit): ")
            query_str = str(query).strip()
            
            if query_str.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            if not query_str:
                print("Please enter a valid query.")
                continue
            
            # For demo, let user specify subject and topic
            print("\nFor better results, specify the subject and topic:")
            subject = input("Subject (medicine/football/technology): ").strip().lower()
            topic = input("Topic (e.g., cardiology, premier league): ").strip().lower()
            
            if not subject or not topic:
                print("Using auto-detection for subject and topic...")
                subject = "general"
                topic = "general"
            
            try:
                print(f"\nProcessing query: '{query_str}'")
                print(f"Subject: {subject}, Topic: {topic}")
                print("-" * 40)
                
                # Execute RAG using DatabaseCrew
                result = database_crew.execute_rag(
                    query=query_str,
                    subject=subject,
                    topic=topic,
                    database_type=database_type,
                    available_databases=available_databases
                )
                
                # Display the result
                if result.get('status') == 'success':
                    guide = result.get('result', {})
                    if isinstance(guide, dict) and 'title' in guide:
                        print("\n" + "="*60)
                        print("GENERATED GUIDE")
                        print("="*60)
                        print(f"\nTitle: {guide.get('title', 'No title')}")
                        print(f"Target Audience: {guide.get('target_audience', 'General')}")
                        print(f"\nIntroduction:\n{guide.get('introduction', 'No introduction')}")
                        
                        sections = guide.get('sections', [])
                        if sections:
                            print(f"\nSections:")
                            for i, section in enumerate(sections, 1):
                                print(f"\n  {i}. {section.get('title', 'No title')}:")
                                desc = section.get('description', 'No description')
                                if len(desc) > 200:
                                    desc = desc[:200] + "..."
                                print(f"     {desc}")
                        
                        print(f"\nConclusion:\n{guide.get('conclusion', 'No conclusion')}")
                    else:
                        print(f"\nResult: {result}")
                else:
                    print(f"\nError: {result.get('message', 'Unknown error')}")
                
            except Exception as e:
                print(f"\nError processing query: {e}")
                import traceback
                traceback.print_exc()
                print("Please try again with a different query.")

def kickoff():
    """Entry point for crewai run command"""
    print("\nStarting DatabaseCrew-based RAG system...")
    main()

if __name__ == "__main__":
    kickoff()

