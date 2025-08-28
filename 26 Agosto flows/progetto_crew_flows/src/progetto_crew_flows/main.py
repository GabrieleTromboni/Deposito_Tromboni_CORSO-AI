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

from crewai.flow.flow import Flow, listen, start, router, or_
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .crews.medical_crew.med_crew import MedicalCrew
from .crews.search_crew.search_crew import SearchCrew
import json
from datetime import datetime

load_dotenv()

# Define models for structured data
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Define our models for structured data
class Section(BaseModel):
    title: str = Field(description="Title of the section")
    description: str = Field(description="Brief description of what the section should cover")

class GuideOutline(BaseModel):
    title: str = Field(description="Title of the guide")
    introduction: str = Field(description="Introduction to the topic")
    target_audience: str = Field(description="Description of the target audience")
    sections: List[Section] = Field(description="List of sections in the guide")
    conclusion: str = Field(description="Conclusion or summary of the guide")

# Define our flow state
class GuideCreatorState(BaseModel):
    query: str = ""
    topic: str = ""
    audience_level: str = ""
    guide_outline: Optional[GuideOutline] = None
    sections_content: Dict[str, str] = {}
    source_type: str = ""  # "RAG" or "WEB_SEARCH"
    sources: List[str] = []
    confidence_score: float = 0.0

class QueryFlow(Flow[GuideCreatorState]):
    # Predefined medical topics available in RAG
    MEDICAL_TOPICS = [
        "cardiology", "neurology", "oncology", "pediatrics", 
        "dermatology", "orthopedics", "psychiatry", "medicine"
    ]
    
    def __init__(self):
        super().__init__()
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """Initialize LLM for topic extraction and validation"""
        return init_chat_model(
            model=CHAT_MODEL,
            api_version="2024-02-15-preview",
            azure_endpoint=CLIENT_AZURE,
            api_key=API_KEY,
            temperature=0.1
        )
    
    @start()
    def extract_topic(self, query: str) -> GuideCreatorState:
        """First step: Extract search topic from user query"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at extracting medical topics from queries. Extract the main medical topic from the user's question."),
            ("human", "Query: {query}\n\nExtract the main medical topic (one word or short phrase):")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        extracted_topic = chain.invoke({"query": query})
        
        # Initialize GuideCreatorState
        state = GuideCreatorState(
            query=query,
            topic=extracted_topic.strip().lower(),
            audience_level="general"  # Default audience level
        )
        
        return state
    
    @listen(extract_topic)
    def validate_topic(self, state: GuideCreatorState) -> GuideCreatorState:
        """Second step: Validate if topic is in allowed list"""
        
        # Check if topic matches any medical topic
        is_medical = any(
            topic in state.topic or state.topic in topic 
            for topic in self.MEDICAL_TOPICS
        )
        
        # Update audience level based on topic type
        state.audience_level = "medical_professional" if is_medical else "general"
        state.confidence_score = 0.95 if is_medical else 0.75
        
        return state
    
    @router(validate_topic)
    def route_to_crew(self, state: GuideCreatorState) -> str:
        """Third/Fourth step: Route to appropriate crew based on validation"""
        
        if state.audience_level == "medical_professional":
            return 'use_RAG'
        else:
            return 'use_WEB_SEARCH'

    @listen('use_RAG')
    def use_RAG(self, state: GuideCreatorState) -> GuideCreatorState:
        """Use medical RAG crew for processing"""
        crew = MedicalCrew()
        result = crew.kickoff(inputs={
            "query": state.query,
            "topic": state.topic
        })
        
        state.source_type = "RAG"
        
        # Parse result into GuideOutline
        state.guide_outline = GuideOutline(
            title=f"Medical Guide: {state.topic.title()}",
            introduction=f"This guide provides medical information about {state.topic}",
            target_audience="Medical professionals and healthcare providers",
            sections=self._parse_sections_from_result(result),
            conclusion="Please consult medical databases for the most current information."
        )
        
        # Extract sources if available
        if hasattr(result, 'sources'):
            state.sources = result.sources
        
        # Add content to sections
        if hasattr(result, 'raw'):
            state.sections_content[state.topic] = result.raw
        
        return state

    @listen('use_WEB_SEARCH')
    def use_web_search(self, state: GuideCreatorState) -> GuideCreatorState:
        """Use web search crew for processing"""
        crew = SearchCrew()
        result = crew.kickoff(inputs={
            "query": state.query,
            "topic": state.topic
        })
        
        state.source_type = "WEB_SEARCH"
        
        # Parse result into GuideOutline
        state.guide_outline = GuideOutline(
            title=f"Information Guide: {state.topic.title()}",
            introduction=f"This guide provides general information about {state.topic}",
            target_audience="General audience",
            sections=self._parse_sections_from_result(result),
            conclusion="Information gathered from web sources. Please verify with authoritative sources."
        )
        
        # Extract sources if available
        if hasattr(result, 'sources'):
            state.sources = result.sources
        
        # Add content to sections
        if hasattr(result, 'raw'):
            state.sections_content[state.topic] = result.raw
            
        return state

    @listen(or_(use_RAG, use_web_search))
    def format_and_save_output(self, state: GuideCreatorState) -> GuideCreatorState:
        """Fifth step: Format and save final output"""
        
        # Save the complete state
        self._save_output(state)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Query: {state.query}")
        print(f"Topic: {state.topic}")
        print(f"Source Type: {state.source_type}")
        print(f"Confidence Score: {state.confidence_score}")
        print(f"Audience Level: {state.audience_level}")
        if state.guide_outline:
            print(f"Guide Title: {state.guide_outline.title}")
            print(f"Number of sections: {len(state.guide_outline.sections)}")
        print("="*60)
        
        return state
    
    def _parse_sections_from_result(self, result) -> List[Section]:
        """Helper method to parse sections from crew result"""
        sections = []
        
        # Default sections if parsing fails
        default_sections = [
            Section(title="Overview", description="General overview of the topic"),
            Section(title="Key Points", description="Main points to understand"),
            Section(title="Additional Information", description="Further details and resources")
        ]
        
        # Try to parse sections from result if it has structure
        if hasattr(result, 'sections'):
            for section in result.sections:
                sections.append(Section(
                    title=section.get('title', 'Section'),
                    description=section.get('description', 'Content')
                ))
        else:
            sections = default_sections
            
        return sections
    
    def _save_output(self, state: GuideCreatorState):
        """Save output to file for tracking"""
        
        filename = f"output_{state.topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_data = state.model_dump()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOutput saved to: {filename}")

def main():
    """Main execution function"""
    flow = QueryFlow()
    
    print("\n" + "="*60)
    print("MEDICAL/GENERAL INFORMATION QUERY SYSTEM")
    print("="*60)
    print("\nAvailable medical topics:", ", ".join(QueryFlow.MEDICAL_TOPICS))
    print("\nYou can ask about medical topics (will use RAG) or general topics (will use web search)")
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
            
            result = flow.kickoff(inputs=query)
            
            # Display the result
            if result.guide_outline:
                print("\n" + "="*60)
                print("GENERATED GUIDE")
                print("="*60)
                print(f"Title: {result.guide_outline.title}")
                print(f"Target Audience: {result.guide_outline.target_audience}")
                print(f"\nIntroduction:\n{result.guide_outline.introduction}")
                print(f"\nSections:")
                for i, section in enumerate(result.guide_outline.sections, 1):
                    print(f"  {i}. {section.title}: {section.description}")
                print(f"\nConclusion:\n{result.guide_outline.conclusion}")
                
                if result.sources:
                    print(f"\nSources:")
                    for source in result.sources:
                        print(f"  - {source}")
        
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try again with a different query.")
    
    # Plot the flow diagram
    print("\nGenerating flow diagram...")
    flow.plot("query_flow_diagram")
    print("Flow diagram saved as 'query_flow_diagram.png'")

if __name__ == "__main__":
    main()

