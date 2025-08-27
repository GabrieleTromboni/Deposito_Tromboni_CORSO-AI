"""
MathCrew - Crew specializzata per operazioni matematiche.
Utilizza un agent mathematician con tool per calcoli numerici.
"""

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, Any, Union, List
import sys
import os

# Aggiungi il percorso tools alla path per importare i custom tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.custom_tool import add_numbers, subtract_numbers, multiply_numbers, divide_numbers

@CrewBase
class MathCrew():
    """
    Crew specializzata in operazioni matematiche e calcoli numerici.
    Utilizza un agent mathematician esperto con accesso a tool matematici.
    """
    
    agents : List[BaseAgent]
    tasks : List[Task]

    @agent
    def mathematician(self) -> Agent:
        """Agent esperto in matematica."""
        return Agent(
            config=self.agents_config['mathematician'],
            tools=[add_numbers, subtract_numbers, multiply_numbers, divide_numbers],
            verbose=True,
            allow_delegation=False,
            max_iter=2,
            memory=True
        )
    
    @task
    def compute_operation(self) -> Task:
        return Task(
            config=self.tasks_config['compute_math_operation']
        )
    
    @task
    def compute_all_operations(self) -> Task:
        return Task(
            config=self.tasks_config['compute_all_math_operations']
        )
    
    @crew
    def crew(self) -> Crew:
        """
        Assembla e restituisce la Crew configurata per operazioni semplici.
        
        Returns:
            Crew: La crew pronta per l'esecuzione con kickoff()
        """
        return Crew(
            agents=[self.agents],
            tasks=[self.tasks], 
            process=Process.sequential,
            verbose=True
        )
        
    def execute(self, a: int, b: int, complex: bool = False) -> Dict[str, Any]:
        """
        Metodo di convenienza per eseguire direttamente il calcolo.
        
        Args:
            a: Il primo numero
            b: Il secondo numero
            complex: Se True, esegue tutte le operazioni matematiche
            
        Returns:
            Dict contenente il risultato del calcolo
        """
        if complex:
            crew = self.complex_crew()
        else:
            crew = self.crew()
            
        result = crew.kickoff(inputs={"a": a, "b": b})
        return {
            "result": result,
            "a": a,
            "b": b,
            "operation": "all operations" if complex else "addition"
        }


# Test standalone della crew
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST STANDALONE - MathCrew")
    print("="*60)
    
    # Crea un'istanza della crew
    math_crew = MathCrew()
    
    # Input di test
    try:
        a = int(input("\nInserisci il primo numero (a): "))
        b = int(input("Inserisci il secondo numero (b): "))
        
        mode = input("\nVuoi eseguire solo la somma o tutte le operazioni? (s/t): ").strip().lower()
        complex_mode = mode == 't'
        
        print(f"\n📍 Test con a={a}, b={b}")
        print(f"📍 Modalità: {'Completa' if complex_mode else 'Solo somma'}")
        print("-"*40)
        
        # Esegui il calcolo
        result = math_crew.execute(a, b, complex=complex_mode)
        print("\n✅ Calcolo completato!")
        print("\n📊 RISULTATO:")
        print("-"*40)
        print(result.get("result"))
        
    except ValueError:
        print("\n❌ Errore: inserire numeri interi validi")
    except Exception as e:
        print(f"\n❌ Errore durante il test: {e}")
