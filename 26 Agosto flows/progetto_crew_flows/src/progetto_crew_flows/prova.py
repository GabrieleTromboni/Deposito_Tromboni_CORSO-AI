import random
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    name: str = ""
    type: str = ""  # "city" or "state"

# Dizionari di esempio per città e stati
CITIES = {
    "Roma": "Roma ospita il Colosseo, uno dei monumenti più famosi al mondo.",
    "Parigi": "Parigi è conosciuta come la città dell'amore.",
    "New York": "New York è chiamata la città che non dorme mai."
}
STATES = {
    "Italia": ["Francia", "Svizzera", "Austria", "Slovenia"],
    "Francia": ["Belgio", "Lussemburgo", "Germania", "Svizzera", "Italia", "Spagna", "Andorra", "Monaco"],
    "Germania": ["Danimarca", "Polonia", "Repubblica Ceca", "Austria", "Svizzera", "Francia", "Lussemburgo", "Belgio", "Paesi Bassi"]
}

class RouterFlow(Flow[ExampleState]):

    @start()
    def start_method(self):
        print("Avvio del flow: scelgo casualmente una città o uno stato")
        if random.choice(["city", "state"]) == "city":
            self.state.type = "city"
            self.state.name = random.choice(list(CITIES.keys()))
        else:
            self.state.type = "state"
            self.state.name = random.choice(list(STATES.keys()))
        print(f"Scelto: {self.state.name} ({self.state.type})")

    @router(start_method)
    def route_method(self):
        if self.state.type == "city":
            return "city"
        else:
            return "state"

    @listen("city")
    def city_fact(self):
        fact = CITIES.get(self.state.name, "Nessun fatto disponibile.")
        print(f"Fatto interessante su {self.state.name}: {fact}")

    @listen("state")
    def state_neighbors(self):
        neighbors = STATES.get(self.state.name, [])
        print(f"Paesi confinanti con {self.state.name}: {', '.join(neighbors)}")

flow = RouterFlow()
flow.plot("my_flow_plot")
flow.kickoff()