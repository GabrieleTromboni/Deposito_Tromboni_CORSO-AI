from crewai.flow.flow import Flow, listen, start, router, or_
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .models import GuideOutline, Section
from .crews.rag_crew.rag_crew import RagCrew
from .crews.search_crew.search_crew import SearchCrew
import json
from datetime import datetime

load_dotenv()

# Define models for structured data
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Define our flow state
class GuideCreatorState(BaseModel):
    query: str = ""
    subject: str = ""  # Added subject field
    topic: str = ""
    audience_level: str = ""
    guide_outline: Optional[GuideOutline] = None
    sections_content: Dict[str, str] = {}
    source_type: str = ""  # "RAG" or "WEB_SEARCH"
    sources: List[str] = []
    confidence_score: float = 0.0
    is_valid_subject: bool = False  # Added validation flag
    is_valid_topic: bool = False    # Added validation flag

class WebRAGFlow(Flow[GuideCreatorState]):
    
    # Predefined subjects and topics available in RAG
    SUBJECTS = {
        'medicine': [
            "cardiology", "neurology", "oncology", "pediatrics", 
            "dermatology", "orthopedics", "psychiatry"
        ],
        'football': [
            "premier league", "la liga", "bundesliga", "serie a"
        ],
        'technology': [
            "artificial intelligence", "machine learning", "blockchain", "cloud computing"
        ]
    }

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
    def extraction(self, query: str) -> GuideCreatorState:
        """First step: Extract subject and topic from user query"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting subjects and topics from queries.
            Available subjects are: {subjects}
            Extract both the main subject category and the specific topic from the user's question.
            Return in format: subject|topic
            If you can't identify a clear subject from the available ones, return: general|topic"""),
            ("human", "Query: {query}\n\nExtract subject and topic (format: subject|topic):")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        extracted = chain.invoke({
            "query": query,
            "subjects": ", ".join(self.SUBJECTS.keys())
        })
        
        # Parse the extraction
        parts = extracted.strip().lower().split('|')
        subject = parts[0] if len(parts) > 0 else "general"
        topic = parts[1] if len(parts) > 1 else parts[0]
        
        # Initialize GuideCreatorState
        state = GuideCreatorState(
            query=query,
            subject=subject,
            topic=topic,
            audience_level="general"  # Default audience level
        )
        
        return state
    
    @listen(extraction)
    def validation(self, state: GuideCreatorState) -> GuideCreatorState:
        """Second step: Validate if subject and topic are in allowed list"""
        
        # Check if subject is valid
        state.is_valid_subject = state.subject in self.SUBJECTS
        
        # Check if topic is valid for the subject
        if state.is_valid_subject:
            state.is_valid_topic = state.topic in self.SUBJECTS[state.subject]
        else:
            state.is_valid_topic = False
        
        # Set confidence score and audience level based on validation
        if state.is_valid_subject and state.is_valid_topic:
            state.confidence_score = 0.95
            state.audience_level = "professional" if state.subject == "medicine" else "specialized"
        else:
            state.confidence_score = 0.75
            state.audience_level = "intermediate"
        
        return state
    
    @router(validation)
    def route_to_crew(self, state: GuideCreatorState) -> str:
        """Third/Fourth step: Route to appropriate crew based on validation"""
        
        if state.is_valid_subject and state.is_valid_topic:
            return 'use_RAG'
        else:
            return 'use_WEB_SEARCH'

    @listen('use_RAG')
    def use_RAG(self, state: GuideCreatorState) -> GuideCreatorState:
        """Use RAG crew for processing"""
        crew = RagCrew()
        result = crew.kickoff(inputs={
            "query": state.query,
            "subject": state.subject,
            "topic": state.topic
        })
        
        state.source_type = "RAG"
        
        # Handle the result based on its type
        if isinstance(result, GuideOutline):
            state.guide_outline = result
        elif isinstance(result, dict):
            # Parse dictionary into GuideOutline
            state.guide_outline = GuideOutline(**result)
        elif isinstance(result, str):
            try:
                # Try to parse JSON string
                import json
                guide_data = json.loads(result)
                state.guide_outline = GuideOutline(**guide_data)
            except:
                # Create a basic outline from string result
                state.guide_outline = GuideOutline(
                    title=f"{state.subject.title()} Guide: {state.topic.title()}",
                    introduction=f"Information about {state.topic} in {state.subject}",
                    target_audience=f"Professionals and specialists in {state.subject}",
                    sections=[
                        Section(title="Retrieved Information", description=str(result)[:500])
                    ],
                    conclusion="Information retrieved from specialized knowledge base."
                )
        else:
            # Fallback for unexpected result types
            state.guide_outline = GuideOutline(
                title=f"{state.subject.title()} Guide: {state.topic.title()}",
                introduction=f"Information about {state.topic} in {state.subject}",
                target_audience=f"Professionals in {state.subject}",
                sections=self._parse_sections_from_result(result),
                conclusion="Information retrieved from knowledge base."
            )
        
        # Add RAG-specific sources
        state.sources = [f"RAG Database - {state.subject}/{state.topic}"]
        
        return state

    @listen('use_WEB_SEARCH')
    def use_web_search(self, state: GuideCreatorState) -> GuideCreatorState:
        """Use web search crew for processing"""
        crew = SearchCrew()
        result = crew.kickoff(inputs={
            "query": state.query,
            "subject": state.subject if state.is_valid_subject else "general",
            "topic": state.topic
        })
        
        state.source_type = "WEB_SEARCH"
        
        # Handle the result based on its type
        if isinstance(result, GuideOutline):
            state.guide_outline = result
        elif isinstance(result, dict):
            state.guide_outline = GuideOutline(**result)
        elif isinstance(result, str):
            try:
                import json
                guide_data = json.loads(result)
                state.guide_outline = GuideOutline(**guide_data)
            except:
                state.guide_outline = GuideOutline(
                    title=f"Information Guide: {state.topic.title()}",
                    introduction=f"Web search results for {state.topic}",
                    target_audience="General audience",
                    sections=[
                        Section(title="Search Results", description=str(result)[:500])
                    ],
                    conclusion="Information gathered from web sources."
                )
        else:
            state.guide_outline = GuideOutline(
                title=f"Information Guide: {state.topic.title()}",
                introduction=f"Information about {state.topic}",
                target_audience="General audience",
                sections=self._parse_sections_from_result(result),
                conclusion="Information from web sources."
            )
        
        # Extract web sources if available
        if hasattr(result, 'sources'):
            state.sources = result.sources
        else:
            state.sources = ["Web search results"]
            
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
        print(f"Subject: {state.subject}")
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
        
        # Default sections
        default_sections = [
            Section(title="Overview", description="General overview of the topic"),
            Section(title="Key Points", description="Main points to understand"),
            Section(title="Additional Information", description="Further details and resources")
        ]
        
        # Try to parse sections from result
        if hasattr(result, 'sections'):
            for section in result.sections:
                if isinstance(section, Section):
                    sections.append(section)
                elif isinstance(section, dict):
                    sections.append(Section(**section))
                else:
                    sections.append(Section(
                        title="Information",
                        description=str(section)[:200]
                    ))
        elif isinstance(result, str):
            # If result is a string, create sections from it
            sections = [
                Section(title="Information", description=result[:500]),
                Section(title="Details", description=result[500:1000] if len(result) > 500 else "Additional information")
            ]
        else:
            sections = default_sections
            
        return sections if sections else default_sections
    
    def _save_output(self, state: GuideCreatorState):
        """Save output to file for tracking"""
        
        filename = f"output_{state.subject}_{state.topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_data = state.model_dump()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nOutput saved to: {filename}")