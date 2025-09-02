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
    def database_retriever(self) -> Agent:
        """Agent specialized in retrieving information from the RAG database"""
        return Agent(
            config=self.agents_config['database_retriever'],
            tools=[retrieve_from_vectordb],
            verbose=True
        )
    
    @agent
    def content_reviewer(self) -> Agent:
        """Agent specialized in synthesizing and formatting retrieved information into guides"""
        return Agent(
            config=self.agents_config['content_reviewer'],
            tools=[format_content_as_guide],
            verbose=True
        )
    
    @task
    def retrieve_info_task(self) -> Task:
        """Task to search the RAG database and retrieve most relevant information"""
        return Task(
            config=self.tasks_config['retrieve_info_task'],
            agent=self.database_retriever(),
        )
    
    @task
    def review_response_task(self) -> Task:
        """Task to create a comprehensive guide from retrieved information"""
        return Task(
            config=self.tasks_config['review_response_task'],
            agent=self.content_reviewer(),
            context=[self.retrieve_info_task()],
            max_execution_time=120  # Add timeout of 2 minutes
        )
    
    @crew
    def crew(self) -> Crew:
        """Create the RAG crew"""
        return Crew(
            agents=[
                self.database_retriever(),
                self.content_reviewer()
            ],
            tasks=[
                self.retrieve_info_task(),
                self.review_response_task()
            ],
            process=Process.sequential,
            verbose=True,
            max_execution_time=180,  # 3 minutes total timeout
            step_callback=self._step_callback
        )
    
    def _step_callback(self, step):
        """Callback to monitor crew execution steps"""
        print(f"🔍 RAG Crew Step: {step}")
        
        # Debug: Print step details if it's a task result
        if hasattr(step, 'description') and hasattr(step, 'output'):
            print(f"   Task: {step.description[:100]}...")
            if step.output:
                output_preview = str(step.output)[:200] if step.output else "No output"
                print(f"   Output preview: {output_preview}...")
        
        return step
    
    def kickoff(self, inputs: dict):
        """Execute the crew with proper input handling"""
        print(f"🚀 RAGCrew.kickoff called with inputs: {inputs}")
        
        # Ensure all required inputs are present
        formatted_inputs = {
            "query": inputs.get("query", ""),
            "topic": inputs.get("topic", ""),
            "subject": inputs.get("subject", "")
        }
        
        print(f"🔧 Formatted inputs for crew: {formatted_inputs}")
        
        try:
            # Execute crew with timeout
            print("⏱️ Starting crew execution...")
            crew_instance = self.crew()
            result = crew_instance.kickoff(inputs=formatted_inputs)
            print(f"✅ Crew execution completed. Result type: {type(result)}")
            
            # Handle CrewOutput - extract the actual JSON content
            json_content = None
            
            if hasattr(result, 'raw'):
                # CrewAI CrewOutput has a 'raw' attribute with the actual content
                json_content = result.raw
                print(f"📄 Extracted raw content from CrewOutput")
            elif hasattr(result, 'content'):
                # Alternative attribute name
                json_content = result.content
                print(f"📄 Extracted content from CrewOutput")
            elif isinstance(result, str):
                # Direct string result
                json_content = result
                print(f"📄 Using direct string result")
            else:
                # Try to convert to string
                json_content = str(result)
                print(f"📄 Converted result to string: {type(result)}")
            
            print(f"📄 JSON content type: {type(json_content)}")
            print(f"📄 JSON content preview: {str(json_content)[:200]}...")
            
            # Parse the JSON content to GuideOutline
            if json_content:
                try:
                    # Clean the JSON content - remove any markdown formatting
                    clean_json = json_content.strip()
                    if clean_json.startswith('```json'):
                        clean_json = clean_json.replace('```json', '').replace('```', '').strip()
                    
                    # Parse as JSON
                    import json
                    guide_data = json.loads(clean_json)
                    
                    # Validate it has the required GuideOutline fields
                    required_fields = ['title', 'introduction', 'target_audience', 'sections', 'conclusion']
                    missing_fields = [field for field in required_fields if field not in guide_data]
                    
                    if missing_fields:
                        print(f"⚠️ Missing required fields in JSON: {missing_fields}")
                        # Add default values for missing fields
                        if 'title' not in guide_data:
                            guide_data['title'] = f"Guide: {inputs.get('topic', 'Unknown Topic')}"
                        if 'introduction' not in guide_data:
                            guide_data['introduction'] = f"Information about {inputs.get('topic', 'the requested topic')}"
                        if 'target_audience' not in guide_data:
                            guide_data['target_audience'] = "General audience"
                        if 'sections' not in guide_data:
                            guide_data['sections'] = []
                        if 'conclusion' not in guide_data:
                            guide_data['conclusion'] = "End of guide"
                    
                    # Create GuideOutline instance
                    final_result = GuideOutline(**guide_data)
                    print("✅ Successfully created GuideOutline from JSON")
                    print(f"📊 Guide title: {final_result.title}")
                    print(f"📊 Guide sections: {len(final_result.sections)}")
                    return final_result
                    
                except json.JSONDecodeError as json_error:
                    print(f"❌ JSON parsing error: {json_error}")
                    print(f"📄 Problematic JSON content: {json_content}")
                except Exception as parse_error:
                    print(f"❌ GuideOutline creation error: {parse_error}")
                    print(f"📄 Guide data: {guide_data if 'guide_data' in locals() else 'Not available'}")
            
            # Fallback if parsing fails
            print("⚠️ Falling back to creating minimal GuideOutline")
            fallback_guide = GuideOutline(
                title=f"Guide: {inputs.get('topic', 'Unknown Topic')}",
                introduction=f"Information about {inputs.get('topic', 'the requested topic')}.",
                target_audience="General audience",
                sections=[],
                conclusion="Guide creation completed with limited information."
            )
            return fallback_guide
                
        except Exception as e:
            print(f"❌ Error during crew execution: {e}")
            # Create a fallback GuideOutline
            fallback_guide = GuideOutline(
                title=f"Guide: {inputs.get('topic', 'Unknown Topic')}",
                introduction=f"Error occurred while processing your query about {inputs.get('topic', 'this topic')}.",
                target_audience="General audience",
                sections=[
                    {
                        "title": "Error Information",
                        "description": f"An error occurred during processing: {str(e)}"
                    }
                ],
                conclusion="Please try again with a different query."
            )
            print("🔄 Returning fallback guide due to error")
            return fallback_guide