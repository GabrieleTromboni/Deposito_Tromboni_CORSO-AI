"""
MathCrew - Crew specializzata per operazioni matematiche.
Utilizza un agent mathematician con tool per calcoli numerici.
"""

from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, Any
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
    
    agents_config : Dict[str, Any]
    tasks_config : Dict[str, Any]

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
        Restituisce una crew di default con task singola operazione.
        Verrà sovrascritta dal metodo kickoff per gestire le modalità dinamiche.
        """
        return Crew(
            agents=[self.mathematician()],
            tasks=[self.compute_math_operation()],
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabilita la memoria per evitare ChromaDB
            cache=False,    # Disabilita anche la cache per evitare problemi
            embedder=None   # Disabilita esplicitamente l'embedder
        )
    
    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """
        Esegue la crew con la configurazione appropriata basata sugli input.
        
        Args:
            inputs: Dizionario contenente:
                - a: primo numero
                - b: secondo numero  
                - mode: 'single' o 'all'
                - operation: tipo di operazione (per mode='single')
        
        Returns:
            Risultato dell'operazione
        """
        mode = inputs.get("mode", "single")
        operation = inputs.get("operation", "add")
        
        # Prepara gli input per la crew
        crew_inputs = {
            "a": inputs["a"],
            "b": inputs["b"]
        }
        
        # Seleziona la task appropriata basandosi sulla modalità
        if mode == "all":
            # Usa la task per tutte le operazioni
            selected_task = self.compute_all_math_operations()
        else:
            # Usa la task per operazione singola
            selected_task = self.compute_math_operation()
            
            # Aggiungi l'operazione specifica agli input per guidare l'agent
            crew_inputs["operation"] = operation
            
            # Mappa delle operazioni con descrizioni dettagliate
            operation_details = {
                "add": {
                    "tool": "add_numbers",
                    "symbol": "+",
                    "name": "addition"
                },
                "subtract": {
                    "tool": "subtract_numbers", 
                    "symbol": "-",
                    "name": "subtraction"
                },
                "multiply": {
                    "tool": "multiply_numbers",
                    "symbol": "*",
                    "name": "multiplication"
                },
                "divide": {
                    "tool": "divide_numbers",
                    "symbol": "/",
                    "name": "division"
                }
            }
            
            if operation in operation_details:
                details = operation_details[operation]
                # Modifica la descrizione della task per essere specifica
                selected_task.description = f"""
                Perform ONLY the {details['name']} operation on numbers {{a}} and {{b}}.
                
                IMPORTANT INSTRUCTIONS:
                1. You MUST use ONLY the {details['tool']} tool
                2. Calculate: {{a}} {details['symbol']} {{b}}
                3. Do NOT use any other mathematical tools
                4. Do NOT perform any other operations
                5. Present only the result of this single operation
                
                The ONLY operation you should perform is: {details['name']}
                """
                
                # Modifica anche l'expected output per essere specifico
                selected_task.expected_output = f"""
                The {details['name']} result:
                - Input numbers: a={{a}}, b={{b}}
                - Operation: {{a}} {details['symbol']} {{b}}
                - Result: [the calculated value]
                - Confirmation that ONLY {details['name']} was performed
                """
        
        # Crea la crew con la task selezionata
        configured_crew = Crew(
            agents=[self.mathematician()],
            tasks=[selected_task],
            process=Process.sequential,
            verbose=True,
            memory=False,  # Disabilita la memoria per evitare ChromaDB
            cache=False,    # Disabilita anche la cache per evitare problemi
            embedder=None   # Disabilita esplicitamente l'embedder
        )
        
        # Esegui la crew con gli input preparati
        return configured_crew.kickoff(inputs=crew_inputs)