import os
from crewai.flow import Flow, start, listen, router
from crewai.flow.flow import or_
from crewai import LLM

# Carica le variabili d'ambiente
from dotenv import load_dotenv
load_dotenv()

# Configura l'LLM con tutti i parametri necessari per Azure
llm = LLM(
    model="azure/gpt-4o",
    api_key=os.getenv("AZURE_API_KEY"),
    base_url=os.getenv("AZURE_API_BASE"),
    api_version=os.getenv("AZURE_API_VERSION")
)

class GeografiaFlow(Flow):

    @start()
    def genera_localita(self):
        """Genera una città o uno stato casuale"""
        try:
            prompt = "Scrivi il nome di una città o di uno stato a caso."
            localita = llm.call(prompt).strip()
            self.state["localita"] = localita
            return {"localita": localita}
        except Exception as e:
            print(f"Errore in genera_localita: {e}")
            # Fallback
            localita = "Roma"
            self.state["localita"] = localita
            return {"localita": localita}

    @listen("genera_localita")
    def classifica_localita(self, result):
        """Determina se la località è una città o uno stato"""
        try:
            localita = result["localita"]
            prompt = f"'{localita}' è una città o uno stato? Rispondi solo con 'città' o 'stato'."
            tipo = llm.call(prompt).lower().strip()
            self.state["tipo"] = tipo
            return {"tipo": tipo, "localita": localita}
        except Exception as e:
            print(f"Errore in classifica_localita: {e}")
            return {"tipo": "città", "localita": result["localita"]}

    @router("classifica_localita")
    def smista(self, result):
        """Instrada in base al tipo"""
        if result["tipo"] == "città" or "città" in result["tipo"]:
            return "fatto_citta"
        elif result["tipo"] == "stato" or "stato" in result["tipo"]:
            return "confini_stato"
        else:
            return "fatto_citta"  # default

    @listen("smista")
    def fatto_citta(self, result):
        if result != "fatto_citta":
            return None
        try:
            citta = self.state["localita"]
            prompt = f"Dimmi un fatto interessante sulla città di {citta}."
            fatto = llm.call(prompt)
            self.state["risultato"] = fatto
            return {"risultato": fatto}
        except Exception as e:
            print(f"Errore in fatto_citta: {e}")
            return {"risultato": f"Fatto interessante su {self.state['localita']}"}

    @listen("smista")
    def confini_stato(self, result):
        if result != "confini_stato":
            return None
        try:
            stato = self.state["localita"]
            prompt = f"Quali paesi confinano con {stato}? Rispondi in elenco."
            confini = llm.call(prompt)
            self.state["risultato"] = confini
            return {"risultato": confini}
        except Exception as e:
            print(f"Errore in confini_stato: {e}")
            return {"risultato": f"Confini di {self.state['localita']}"}

    @listen(or_("fatto_citta", "confini_stato"))
    def risultato_finale(self, result):
        """Step finale che raccoglie i due branch"""
        localita = self.state.get("localita")
        output = self.state.get("risultato")
        return f"Località: {localita}\nRisultato: {output}"

if __name__ == "__main__":
    
    """Run the geografia flow"""
    flow = GeografiaFlow()
    flow.kickoff()
    print("\n=== Flow Complete ===")
    """Generate a visualization of the flow"""
    flow.plot("geografia_flow")
    print("Flow visualization saved to geografia_flow.html")
