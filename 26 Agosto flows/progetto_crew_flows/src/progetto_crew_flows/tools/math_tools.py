from typing import Type, Dict, Any, Union, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai.tools import BaseTool, tool

# Math Tools
@tool
def add_numbers(a: int, b: int) -> int:
    """
    add_numbers(a: int, b: int) -> int
    Restituisce la somma di due numeri interi passati come parametri.
    Parametri:
      - a: primo addendo (int)
      - b: secondo addendo (int)
    Ritorno:
      - int: somma di a e b
    """
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """
    Moltiplica due numeri interi.

    Args:
      - a: primo fattore (int)
      - b: secondo fattore (int)

    Returns:
      - int: prodotto di a e b
    """
    return a * b

@tool
def subtract_numbers(a: int, b: int) -> int:
    """
    Sottrae il secondo numero dal primo.
    
    Args:
        a: Il minuendo
        b: Il sottraendo
    
    Returns:
        La differenza tra a e b
    """
    return a - b

@tool
def divide_numbers(a: int, b: int) -> Union[float, str]:
    """
    Divide il primo numero per il secondo.
    
    Args:
        a: Il dividendo
        b: Il divisore
    
    Returns:
        Il quoziente di a/b o un messaggio di errore se b=0
    """
    if b == 0:
        return "Errore: Divisione per zero non permessa"
    return a / b