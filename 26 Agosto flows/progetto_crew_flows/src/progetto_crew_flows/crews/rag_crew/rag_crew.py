from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from progetto_crew_flows.tools.rag_tool import retrieve_from_vectordb, format_content_as_guide
from progetto_crew_flows.models import GuideOutline
import json
from typing import List

@CrewBase
class RAGCrew():
    """RAG crew for information retrieval from vector database"""
    
    agents_config : List[BaseAgent]
    tasks_config : List[Task]
        
    @agent
    def rag_retriever(self) -> Agent:
        """Agent specialized in retrieving information from the RAG database"""
        return Agent(
            config=self.agents_config['database_retriever'],
            tools=[retrieve_from_vectordb],
            verbose=True
        )
    
    @agent
    def content_synthesizer(self) -> Agent:
        """Agent specialized in synthesizing and formatting retrieved information into guides"""
        return Agent(
            config=self.agents_config['content_reviewer'],
            tools=[format_content_as_guide],
            verbose=True
        )
    
    @task
    def search_rag_database(self) -> Task:
        """Task to search the RAG database and retrieve most relevant information"""
        return Task(
            config=self.tasks_config['retrieve_info_task'],
            agent=self.rag_retriever()
        )
    
    @task
    def create_guide_from_rag(self) -> Task:
        """Task to create a comprehensive guide from retrieved information"""
        return Task(
            config=self.tasks_config['review_response_task'],
            agent=self.content_synthesizer(),
            context=[self.search_rag_database()],
            output_pydantic=GuideOutline
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the RAG crew"""
        return Crew(
            agents=[
                self.rag_retriever(),
                self.content_synthesizer()
            ],
            tasks=[
                self.search_rag_database(),
                self.create_guide_from_rag()
            ],
            process=Process.sequential,
            verbose=True
        )
    
    def kickoff(self, inputs: dict):
        """Execute the crew with proper input handling"""
        # Ensure all required inputs are present
        formatted_inputs = {
            "query": inputs.get("query", ""),
            "topic": inputs.get("topic", ""),
            "subject": inputs.get("subject", "")
        }
        
        # Execute crew
        result = self.crew().kickoff(inputs=formatted_inputs)
        
        # Parse result to ensure it's a GuideOutline
        if isinstance(result, str):
            try:
                # Try to parse as JSON
                guide_data = json.loads(result)
                return GuideOutline(**guide_data)
            except:
                # Return result as is
                return result
        
        return result