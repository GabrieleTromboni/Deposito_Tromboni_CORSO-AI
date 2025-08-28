"""
Custom DuckDuckGo Search Tool for CrewAI
"""

from crewai_tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
import json

class SearchInput(BaseModel):
    """Input schema for DuckDuckGo search."""
    query: str = Field(description="The search query to execute")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class CustomDuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search the web using DuckDuckGo for information on any topic"
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """
        Execute a DuckDuckGo search and return formatted results.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Formatted string with search results
        """
        try:
            with DDGS(verify=False) as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                
                if not results:
                    return f"No results found for query: {query}"
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append(
                        f"{i}. **{result.get('title', 'No title')}**\n"
                        f"   URL: {result.get('href', 'No URL')}\n"
                        f"   Summary: {result.get('body', 'No summary available')}\n"
                    )
                
                return "\n".join(formatted_results)
                
        except Exception as e:
            return f"Error during search: {str(e)}"