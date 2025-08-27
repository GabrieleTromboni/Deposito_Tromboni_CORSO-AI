#!/usr/bin/env python
"""
Main Flow per orchestrare le diverse Crew.
Questo file contiene solo il Flow principale che coordina l'esecuzione
delle Crew definite nei file separati.
"""

# Importazioni standard
import sys
from typing import Any, Dict

# Importazioni CrewAI
from crewai.flow.flow import Flow, start, listen, router

# Importa le Crew personalizzate dai file di configurazione
from crews.search_crew import SearchCrew  # Crew per ricerca web
from crews.math_crew import MathCrew      # Crew per operazioni matematiche

# ============================================================================
# FLOW PRINCIPALE - Orchestrazione delle Crew
# ============================================================================

class AgentFlow(Flow):
    """
    Flow principale che gestisce l'intero processo:
    1. Riceve l'operazione da eseguire tramite inputs
    2. Instrada alla Crew appropriata tramite il router
    3. Richiede e Raccoglie gli input necessari
    4. Esegue la Crew selezionata
    5. Restituisce i risultati in base all'operazione effettuata
    """
    
    # ========================================================================
    # STEP 1: Punto di ingresso del Flow
    # ========================================================================
    @start()
    def get_operation(self) -> str:
        """
        Punto di partenza del Flow.
        Chiede all'utente quale operazione desidera eseguire.
        
        Returns:
            str: L'operazione scelta ('ricerca' o 'matematica')
        """
        print("\n" + "="*60)
        print("🚀 FLOW AVVIATO - Sistema di Orchestrazione Crew")
        print("="*60)
        
        # Menu delle opzioni disponibili
        print("\nOperazioni al momento disponibili:")
        print("  📌 'ricerca'    - Esegue una ricerca web")
        print("  📌 'matematica' - Esegue operazioni matematiche")
        
        # Raccolta input utente
        operation = input("\n➜ Scegli un'operazione: ").strip().lower()
        
        # Salva l'operazione nello state del Flow per uso futuro
        self.state["operation"] = operation
        
        return operation

    # ========================================================================
    # STEP 2: Router per instradamento dinamico
    # ========================================================================
    @router(get_operation)
    def route_operation(self) -> str:
        """
        Router che decide quale percorso seguire basandosi sull'operazione scelta.
        Questo metodo viene chiamato automaticamente dopo get_operation.
        
        Returns:
            str: Il nome del prossimo step da eseguire
        """
        operation = self.state.get("operation", "")
        
        # Mappa le operazioni ai rispettivi handler
        routing_map = {
            "ricerca": "get_search_input",
            "matematica": "get_math_input"
        }
        
        # Instrada all'handler appropriato o gestisce input non validi
        if operation in routing_map:
            print(f"\n✓ Operazione '{operation}' riconosciuta")
            return routing_map[operation]
        else:
            print(f"\n✗ Operazione '{operation}' non riconosciuta")
            return "invalid_operation"
    
    # ========================================================================
    # PERCORSO RICERCA: Input e esecuzione
    # ========================================================================
    
    @listen("get_search_input")
    def get_search_query(self) -> str:
        """
        Raccoglie la query di ricerca dall'utente.
        Attivato quando l'utente sceglie 'ricerca'.
        
        Returns:
            str: Nome del prossimo step ('perform_search')
        """
        print("\n" + "-"*40)
        print("📝 CONFIGURAZIONE RICERCA")
        print("-"*40)
        
        # Richiesta query di ricerca
        query = input("➜ Inserisci la tua query di ricerca: ").strip()
        
        # Validazione input
        if not query:
            print("⚠️  Query vuota, uso query di default")
            query = "artificial intelligence latest news"
        
        # Salva la query nello state
        self.state["query"] = query
        print(f"✓ Query salvata: '{query}'")
        
        return "perform_search"
    
    @listen("perform_search")
    def execute_search(self) -> str:
        """
        Esegue la SearchCrew con la query fornita.
        Utilizza la Crew definita in crews/search_crew.py
        
        Returns:
            str: 'completed' se successo, 'error' se fallimento
        """
        print("\n" + "-"*40)
        print("🔍 ESECUZIONE RICERCA WEB")
        print("-"*40)
        
        # Recupera la query dallo state
        query = self.state.get("query", "")
        print(f"➜ Avvio SearchCrew per: '{query}'")
        
        try:
            # Inizializza la SearchCrew
            search_crew = SearchCrew().crew()
            
            # Esegue la Crew con gli input
            print("⏳ Ricerca in corso...")
            result = search_crew.kickoff(inputs={"query": query})
            
            # Estrai e processa il risultato
            if hasattr(result, 'raw'):
                output = result.raw
            elif isinstance(result, dict):
                output = result.get('result', str(result))
            else:
                output = str(result)
            
            # Salva il risultato nello state
            self.state["result"] = output
            
            # Mostra il risultato
            print("\n" + "="*60)
            print("📊 RISULTATO RICERCA")
            print("="*60)
            print(f"Query: {query}")
            print("-"*60)
            print(output)
            print("="*60)
            
            return "completed"
            
        except Exception as e:
            print(f"\n❌ Errore durante la ricerca: {e}")
            self.state["error"] = str(e)
            return "error"
    
    # ========================================================================
    # PERCORSO MATEMATICA: Input e esecuzione
    # ========================================================================
    
    @listen("get_math_input")
    def get_numbers(self) -> str:
        """
        Raccoglie i numeri per l'operazione matematica.
        Attivato quando l'utente sceglie 'matematica'.
        
        Returns:
            str: 'perform_math' se input validi, 'error' altrimenti
        """
        print("\n" + "-"*40)
        print("📝 CONFIGURAZIONE OPERAZIONE MATEMATICA")
        print("-"*40)
        
        try:
            # Richiesta primo numero
            a_str = input("➜ Inserisci il primo numero (a): ").strip()
            a = int(a_str)
            print(f"✓ Primo numero: {a}")
            
            # Richiesta secondo numero
            b_str = input("➜ Inserisci il secondo numero (b): ").strip()
            b = int(b_str)
            print(f"✓ Secondo numero: {b}")
            
            # Salva i numeri nello state
            self.state["a"] = a
            self.state["b"] = b
            
            return "perform_math"
            
        except ValueError as e:
            print(f"❌ Input non valido: devono essere numeri interi")
            self.state["error"] = "Input non valido"
            return "error"
    
    @listen("perform_math")
    def execute_math(self) -> str:
        """
        Esegue la MathCrew con i numeri forniti.
        Utilizza la Crew definita in crews/math_crew.py
        
        Returns:
            str: 'completed' se successo, 'error' se fallimento
        """
        print("\n" + "-"*40)
        print("🧮 ESECUZIONE CALCOLO MATEMATICO")
        print("-"*40)
        
        # Recupera i numeri dallo state
        a = self.state.get("a", 0)
        b = self.state.get("b", 0)
        print(f"➜ Avvio MathCrew per: {a} + {b}")
        
        try:
            # Inizializza la MathCrew
            math_crew = MathCrew().crew()
            
            # Esegue la Crew con gli input
            print("⏳ Calcolo in corso...")
            result = math_crew.kickoff(inputs={"a": a, "b": b})
            
            # Estrai e processa il risultato
            if hasattr(result, 'raw'):
                output = result.raw
            elif isinstance(result, dict):
                output = result.get('result', str(result))
            else:
                output = str(result)
            
            # Salva il risultato nello state
            self.state["result"] = output
            
            # Mostra il risultato
            print("\n" + "="*60)
            print("📊 RISULTATO CALCOLO")
            print("="*60)
            print(f"Operazione: {a} + {b}")
            print(f"Risultato: {output}")
            print("="*60)
            
            return "completed"
            
        except Exception as e:
            print(f"\n❌ Errore durante il calcolo: {e}")
            self.state["error"] = str(e)
            return "error"
    
    # ========================================================================
    # GESTIONE ERRORI E COMPLETAMENTO
    # ========================================================================
    
    @listen("invalid_operation")
    def handle_invalid(self) -> str:
        """
        Gestisce operazioni non valide o non riconosciute.
        
        Returns:
            str: Sempre 'error'
        """
        operation = self.state.get("operation", "unknown")
        print("\n" + "="*60)
        print("❌ OPERAZIONE NON VALIDA")
        print("="*60)
        print(f"L'operazione '{operation}' non è riconosciuta.")
        print("Operazioni valide: 'ricerca', 'matematica'")
        
        self.state["error"] = f"Operazione '{operation}' non valida"
        return "error"
    
    @listen("completed")
    def show_completion(self) -> str:
        """
        Mostra il messaggio di completamento con successo.
        
        Returns:
            str: Sempre 'end'
        """
        print("\n" + "="*60)
        print("✅ FLOW COMPLETATO CON SUCCESSO")
        print("="*60)
        
        # Mostra un riepilogo se disponibile
        if "operation" in self.state:
            print(f"Operazione eseguita: {self.state['operation']}")
        if "result" in self.state:
            print(f"Risultato salvato nello state")
        
        return "end"
    
    @listen("error")
    def handle_error(self) -> str:
        """
        Gestisce tutti gli errori del Flow.
        
        Returns:
            str: Sempre 'end'
        """
        error = self.state.get("error", "Errore sconosciuto")
        print("\n" + "="*60)
        print("❌ FLOW TERMINATO CON ERRORE")
        print("="*60)
        print(f"Dettaglio errore: {error}")
        
        return "end"
    
    @listen("end")
    def end_flow(self) -> None:
        """
        Punto finale del Flow.
        Esegue pulizia e mostra messaggio di chiusura.
        """
        print("\n" + "="*60)
        print("🏁 FLOW TERMINATO")
        print("="*60)
        print("Grazie per aver utilizzato il sistema di orchestrazione Crew!")
        print("="*60 + "\n")

# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    """
    Funzione principale che gestisce il ciclo di esecuzione del Flow.
    Permette multiple esecuzioni fino a quando l'utente decide di uscire.
    """
    # Banner iniziale
    print("\n" + "="*70)
    print(" "*20 + "🤖 SISTEMA AGENT FLOW CON CREWAI 🤖")
    print("="*70)
    print("Benvenuto nel sistema di orchestrazione intelligente delle Crew!")
    print("Questo sistema può eseguire ricerche web e operazioni matematiche.")
    print("="*70)
    
    # Inizializza il Flow una sola volta
    flow = AgentFlow()
    
    # Ciclo principale di esecuzione
    while True:
        try:
            # Esegui il flow
            print("\n" + "▶"*30)
            result = flow.kickoff()
            print("◀"*30)
            
            # Chiedi se l'utente vuole continuare
            print("\n" + "-"*50)
            cont = input("🔄 Vuoi eseguire un'altra operazione? (s/n): ").strip().lower()
            
            if cont not in ("s", "si", "y", "yes", "sì"):
                print("\n" + "="*50)
                print(" "*15 + "👋 ARRIVEDERCI!")
                print("="*50)
                print("Grazie per aver utilizzato il sistema Agent Flow!")
                print("="*50 + "\n")
                break
            
            # Reset dello state per la prossima esecuzione
            flow.state = {}
            print("\n✓ State resettato per nuova operazione")
            
        except KeyboardInterrupt:
            # Gestione interruzione da tastiera (Ctrl+C)
            print("\n\n" + "="*50)
            print(" "*10 + "⚠️  INTERRUZIONE UTENTE RILEVATA")
            print("="*50)
            print("Il programma è stato interrotto dall'utente.")
            print("="*50 + "\n")
            break
            
        except Exception as e:
            # Gestione errori imprevisti
            print(f"\n" + "="*50)
            print(" "*15 + "❌ ERRORE CRITICO")
            print("="*50)
            print(f"Si è verificato un errore imprevisto: {e}")
            print("="*50)
            
            retry = input("\n🔄 Vuoi riprovare? (s/n): ").strip().lower()
            if retry not in ("s", "si", "y", "yes", "sì"):
                print("\n👋 Chiusura del programma...")
                break
            
            # Reset dello state in caso di errore
            flow.state = {}

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()