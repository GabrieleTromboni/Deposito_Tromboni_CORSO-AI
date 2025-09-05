
import requests
import urllib3
# Disabilita SSL Warning (solo per test)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    url = "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41&hourly=temperature_2m"
    response = requests.get(url, verify=False)
    print("✅ Richiesta effettuata con successo!")
    print("Status Code:", response.status_code)
    risposta = response.json()
    print("Risposta JSON:", risposta)
    print("Alle ore:", risposta["hourly"]["time"][0])
    print("Temperatura:", risposta["hourly"]["temperature_2m"][0])


main()
