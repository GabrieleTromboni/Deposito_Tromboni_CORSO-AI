from crewai.flow.flow import Flow, listen, start, router, or_
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from progetto_crew_flows.models import GuideOutline, Section
from progetto_crew_flows.crews.rag_crew.rag_crew import RAGCrew
from progetto_crew_flows.crews.search_crew.search_crew import SearchCrew
import json
from datetime import datetime

load_dotenv()

# Define models for structured data
API_KEY = os.getenv("AZURE_API_KEY")
CLIENT_AZURE = os.getenv("AZURE_API_BASE")
API_VERSION = os.getenv("AZURE_API_VERSION")
# Support both CHAT_MODEL and legacy MODEL env vars for chat deployment
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("MODEL")

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
    found_subjects: List[str] = []  # Additional found subjects
    found_topics: List[str] = []    # Additional found topics

class WebRAGFlow(Flow[GuideCreatorState]):
    
    # Predefined subjects and topics available in RAG
    SUBJECTS = {
        'medicine': [
            "cardiology", "neurology", "psychiatry"
        ],
        'football': [
            "premier league", "serie a"
        ],
        'technology': [
            "artificial intelligence", "blockchain"
        ]
    }

    def __init__(self):
        super().__init__()
        self.llm = self._init_llm()
        self.query_input = None  # Store query input here
        
    def kickoff(self, inputs=None, **kwargs):
        """Override kickoff to capture inputs before starting flow"""
        if inputs and isinstance(inputs, dict):
            self.query_input = inputs.get('query')
        print(f"🔧 WebRAGFlow kickoff called with query: {self.query_input}")
        return super().kickoff(inputs=inputs, **kwargs)
        
    def _init_llm(self):
        """Initialize LLM for topic extraction and validation"""
        return AzureChatOpenAI(
            deployment_name=CHAT_MODEL,
            openai_api_version=API_VERSION,
            azure_endpoint=CLIENT_AZURE,
            openai_api_key=API_KEY,
            temperature=0.1
        )
    
    @start()
    def extraction(self) -> GuideCreatorState:
        """First step: Extract subject and topic from user query"""
        
        # Use the query stored during kickoff
        query = self.query_input
        
        if query is None:
            # Try backup methods to get the query
            inputs = getattr(self, '_inputs', {})
            print(f"🔍 Flow inputs: {inputs}")
            
            if isinstance(inputs, dict):
                query = inputs.get('query')
        
        if query is None:
            raise ValueError(f"No query provided to the flow. Query input: {self.query_input}")
        
        print(f"🔍 Extracting from query: {query}")
        print(f"🔍 Query type: {type(query)}")
        print(f"🔍 Query length: {len(str(query))}")
        
        print(f"🔍 Processing query: {query}")
        
        # Check if any subjects or topics from SUBJECTS are present in the query
        query_lower = query.lower()
        found_subjects = []
        found_topics = []
        
        print(f"🔍 Searching for subjects and topics in query...")
        
        # Search for subjects in the query
        for subject in self.SUBJECTS.keys():
            if subject.lower() in query_lower:
                found_subjects.append(subject)
                print(f"   ✓ Found subject: {subject}")
        
        # Search for topics in the query
        for subject, topics in self.SUBJECTS.items():
            for topic in topics:
                if topic.lower() in query_lower:
                    found_topics.append(topic)
                    found_subjects.append(subject)  # Also add the parent subject
                    print(f"   ✓ Found topic: {topic} (subject: {subject})")
        
        # Remove duplicates
        found_subjects = list(set(found_subjects))
        found_topics = list(set(found_topics))
        
        print(f"🔍 Summary - Found subjects: {found_subjects}, Found topics: {found_topics}")
        
        # Determine primary subject and topic for processing
        if found_subjects:
            # Use the first found subject
            primary_subject = found_subjects[0]
            print(f"🔍 Primary subject: {primary_subject}")
        else:
            # Fallback to LLM extraction if no direct matches
            print(f"🔍 No direct subject matches found, using LLM extraction...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting subjects and topics from queries.
                Available subjects are: {subjects}
                Extract the main subject category from the user's question.
                Return only the subject name, or 'general' if no clear match."""),
                ("human", "Query: {query}\n\nExtract subject:")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            primary_subject = chain.invoke({
                "query": query,
                "subjects": ", ".join(self.SUBJECTS.keys())
            }).strip().lower()
        
        # Determine primary topic
        if found_topics:
            # Use the first found topic
            primary_topic = found_topics[0]
            print(f"🔍 Primary topic: {primary_topic}")
        else:
            # Fallback to LLM extraction
            print(f"🔍 No direct topic matches found, using LLM extraction...")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at extracting topics from queries.
                Extract the main topic or focus from the user's question.
                Return a concise topic description."""),
                ("human", "Query: {query}\n\nExtract main topic:")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            primary_topic = chain.invoke({"query": query}).strip().lower()
        
        # Initialize GuideCreatorState with additional metadata
        state = GuideCreatorState(
            query=query,
            subject=primary_subject,
            topic=primary_topic,
            audience_level="general",  # Default audience level
            found_subjects=found_subjects if found_subjects else [],
            found_topics=found_topics if found_topics else []
        )
        
        print(f"🔍 State created - Subject: {state.subject}, Topic: {state.topic}")
        print(f"🔍 Additional found items - Subjects: {state.found_subjects}, Topics: {state.found_topics}")
        
        return state
    
    @listen(extraction)
    def validation(self, state: GuideCreatorState) -> GuideCreatorState:
        """Second step: Validate if subject and topic are in allowed list"""
        
        print(f"🔍 Validation - Primary subject: {state.subject}, Primary topic: {state.topic}")
        print(f"🔍 Validation - Found subjects: {state.found_subjects}, Found topics: {state.found_topics}")
        
        # Check if primary subject is valid
        state.is_valid_subject = state.subject in self.SUBJECTS.keys()
        
        # Check if primary topic is valid for the subject
        if state.is_valid_subject:
            state.is_valid_topic = state.topic in self.SUBJECTS[state.subject]
        else:
            state.is_valid_topic = False
        
        # Enhanced validation: if we found ANY subjects/topics in the query, consider it valid for RAG
        has_rag_content = False
        
        # Check if any found subjects are valid
        for found_subject in state.found_subjects:
            if found_subject in self.SUBJECTS.keys():
                has_rag_content = True
                print(f"✓ Found valid subject in query: {found_subject}")
                break
        
        # Check if any found topics are valid
        if not has_rag_content:
            for found_topic in state.found_topics:
                for subject, topics in self.SUBJECTS.items():
                    if found_topic in topics:
                        has_rag_content = True
                        print(f"✓ Found valid topic in query: {found_topic} (subject: {subject})")
                        # Update primary subject if we found a valid topic
                        if not state.is_valid_subject:
                            state.subject = subject
                            state.is_valid_subject = True
                        if not state.is_valid_topic:
                            state.topic = found_topic
                            state.is_valid_topic = True
                        break
                if has_rag_content:
                    break
        
        # Override validation if we found RAG content
        if has_rag_content:
            print(f"🔍 RAG content detected in query - using RAG route")
            state.is_valid_subject = True
            state.is_valid_topic = True
        
        print(f"🔍 Final validation - Subject valid: {state.is_valid_subject}, Topic valid: {state.is_valid_topic}")
        print(f"🔍 Final state - Subject: {state.subject}, Topic: {state.topic}")
        
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
        crew = RAGCrew()
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