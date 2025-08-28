"""
MathCrew - Crew specializzata per operazioni matematiche.
Utilizza un agent mathematician con tool per calcoli numerici.
"""

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, Any, List
import sys
import os

# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.custom_tool import add_numbers, subtract_numbers, multiply_numbers, divide_numbers

@CrewBase
class MathCrew():
    """
    Crew specializzata in operazioni matematiche e calcoli numerici.
    Utilizza un singolo agent mathematician esperto con accesso a tutti i tool matematici.
    Può operare in due modalità:
    - Singola operazione: esegue solo l'operazione richiesta tramite compute_math_operation
    - Tutte le operazioni: esegue tutte le operazioni disponibili tramite compute_all_math_operations
    """
    
    agents_config : List[BaseAgent]
    tasks_config : List[Task]

    @agent
    def mathematician(self) -> Agent:
        """Agent esperto in matematica con tutti i tool disponibili."""
        return Agent(
            config=self.agents_config['mathematician'],
            tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            memory=False
        )
    
    @task
    def compute_math_operation(self) -> Task:
        """Task per eseguire una singola operazione matematica (definita in tasks.yaml)."""
        return Task(
            config=self.tasks_config['compute_math_operation'],
            agent=self.mathematician()
        )
    
    @task
    def compute_all_math_operations(self) -> Task:
        """Task per eseguire tutte le operazioni matematiche (definita in tasks.yaml)."""
        return Task(
            config=self.tasks_config['compute_all_math_operations'],
            agent=self.mathematician()
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Restituisce una crew di default basata sulle task da fare.
        Esegue la crew con la configurazione appropriata basata sugli input.
        """
        return Crew(
            agents=self.agents_config,
            tasks=self.tasks_config,
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabilita la memoria per evitare ChromaDB
            cache=False,    # Disabilita anche la cache per evitare problemi
            embedder=None   # Disabilita esplicitamente l'embedder
        )