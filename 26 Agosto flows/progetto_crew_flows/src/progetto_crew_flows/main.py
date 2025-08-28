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

from progetto_crew_flows.crews.database_crew.data_crew import DatabaseCrew
from WebRAG_flow import WebRAGFlow

def main():
    """Main execution function"""
    
    # First step: Create/Update RAG database
    print("\n" + "="*60)
    print("INITIALIZING RAG DATABASE")
    print("="*60)

    # Initialize DatabaseCrew to create database with all subjects
    database_crew = DatabaseCrew()

    # Initialize database with all subjects and topics
    print("\nInitializing database with subjects and topics...")
    initialization_result = database_crew.kickoff(WebRAGFlow.SUBJECTS)
    print(f"Result: {initialization_result}")
    
    # Initialize the flow
    flow = WebRAGFlow()
    
    print("\n" + "="*60)
    print("INFORMATION QUERY SYSTEM")
    print("="*60)
    print("\nAvailable subjects and topics:")
    for subject, topics in WebRAGFlow.SUBJECTS.items():
        print(f"\n{subject.upper()}:")
        for topic in topics:
            print(f"  - {topic}")
    
    print("\n" + "-"*60)
    print("You can ask questions about any of the above subjects and topics.")
    print("If your question matches our knowledge base, RAG will be used.")
    print("Otherwise, web search will be performed.")
    print("Type 'exit' to quit\n")
    
    while True:
        # Get user input
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        
        if query.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            # Process the query
            print(f"\nProcessing query: '{query}'")
            print("-" * 40)
            
            result = flow.kickoff(query)
            
            # Display processing details
            print(f"\nExtracted Subject: {result.subject}")
            print(f"Extracted Topic: {result.topic}")
            print(f"Validation: Subject={'✓' if result.is_valid_subject else '✗'}, Topic={'✓' if result.is_valid_topic else '✗'}")
            print(f"Source Type: {result.source_type}")
            print(f"Confidence Score: {result.confidence_score:.2f}")
            
            # Display the result
            if result.guide_outline:
                print("\n" + "="*60)
                print("GENERATED GUIDE")
                print("="*60)
                print(f"\nTitle: {result.guide_outline.title}")
                print(f"Target Audience: {result.guide_outline.target_audience}")
                print(f"\nIntroduction:\n{result.guide_outline.introduction}")
                print(f"\nSections:")
                for i, section in enumerate(result.guide_outline.sections, 1):
                    print(f"\n  {i}. {section.title}:")
                    # Truncate long descriptions for display
                    desc = section.description
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    print(f"     {desc}")
                print(f"\nConclusion:\n{result.guide_outline.conclusion}")
                
                if result.sources:
                    print(f"\nSources:")
                    for source in result.sources:
                        print(f"  - {source}")
            else:
                print("\nNo guide could be generated. Please try a different query.")
        
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()
            print("Please try again with a different query.")
    
    # Plot the flow diagram
    print("\nGenerating flow diagram...")
    try:
        flow.plot("query_flow_diagram")
        print("Flow diagram saved as 'query_flow_diagram.png'")
    except Exception as e:
        print(f"Could not generate flow diagram: {e}")

if __name__ == "__main__":
    main()

