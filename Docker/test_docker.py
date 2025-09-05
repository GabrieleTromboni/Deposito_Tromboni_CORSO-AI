"""
Modulo per ottenere previsioni meteorologiche tramite API Open-Meteo.
Supporta la containerizzazione con Docker.
"""

import requests
import urllib3

# Disabilita SSL Warning (solo per test)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DOCKERFILE_CONTENT = '''
# Usa un'immagine ufficiale di Python come base
FROM python:latest

# Crea e imposta la directory di lavoro nel container
WORKDIR /app

# Copia i file requirements.txt
COPY requirements_simple.txt ./requirements.txt

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia il file dell'app nel container
COPY test_docker.py .

# Comando che Docker esegue quando il container parte
CMD ["python", "test_docker.py"]
'''


def get_city_coordinates(city_name):
    """Ottiene le coordinate geografiche di una città usando l'API di geocoding di Open-Meteo"""
    geocoding_url = (f"https://geocoding-api.open-meteo.com/v1/search?"
                     f"name={city_name}&count=1&language=it")

    try:
        response = requests.get(geocoding_url, verify=False, timeout=10)
        response.raise_for_status()

        data = response.json()

        if not data.get('results'):
            print(f"❌ Città '{city_name}' non trovata!")
            return None, None, None

        result = data['results'][0]
        latitude = result['latitude']
        longitude = result['longitude']
        country = result.get('country', 'N/A')

        print(f"📍 Città trovata: {result['name']}, {country}")
        print(f"🌍 Coordinate: Lat {latitude}, Lon {longitude}")

        return latitude, longitude, result['name']

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nella richiesta di geocoding: {e}")
        return None, None, None


def get_user_preferences():
    """Ottiene le preferenze dell'utente per la visualizzazione delle previsioni"""
    print("\n⚙️  CONFIGURAZIONE PREVISIONI")
    print("=" * 40)

    # Scelta del periodo
    print("\n📅 Per quanti giorni vuoi le previsioni?")
    print("1️⃣  1 giorno")
    print("3️⃣  3 giorni")
    print("7️⃣  7 giorni")

    try:
        days_choice = input("\nScegli (1/3/7): ").strip()
        if days_choice in ['1', '3', '7']:
            days = int(days_choice)
        else:
            print("⚠️  Scelta non valida, uso 3 giorni come predefinito")
            days = 3
    except (EOFError, ValueError):
        days = 3
        print("⚠️  Usando 3 giorni come predefinito")

    # Scelta dei parametri da visualizzare
    print("\n🌡️  Quali parametri vuoi visualizzare?")
    print("1️⃣  Solo temperature")
    print("2️⃣  Temperature + Velocità vento")
    print("3️⃣  Temperature + Precipitazioni")
    print("4️⃣  Tutti i parametri (Temperature + Vento + Precipitazioni)")

    try:
        param_choice = input("\nScegli (1/2/3/4): ").strip()
        if param_choice not in ['1', '2', '3', '4']:
            print("⚠️  Scelta non valida, mostro tutti i parametri")
            param_choice = '4'
    except EOFError:
        param_choice = '4'
        print("⚠️  Mostrando tutti i parametri")

    # Converti la scelta in parametri booleani
    show_temp = True  # Temperature sempre mostrate
    show_wind = param_choice in ['2', '4']
    show_precip = param_choice in ['3', '4']

    print("\n✅ Configurazione scelta:")
    print(f"📅 Giorni: {days}")
    print("🌡️  Temperature: ✅")
    print(f"💨 Velocità vento: {'✅' if show_wind else '❌'}")
    print(f"🌧️  Precipitazioni: {'✅' if show_precip else '❌'}")

    return days, show_temp, show_wind, show_precip


def display_hourly_forecast(hourly_data, show_temp, show_wind, show_precip):
    """Mostra le previsioni orarie (funzione helper)"""
    print("\n📊 PREVISIONI ORARIE (prime 12 ore di oggi):")
    print("-" * 70)

    for i in range(min(12, len(hourly_data['time']))):
        time = hourly_data['time'][i]
        hour = time.split('T')[1][:5]
        output_line = f"🕐 {hour}"

        if show_temp:
            temp = hourly_data['temperature_2m'][i]
            output_line += f" | 🌡️  {temp:.1f}°C"

        if show_precip:
            precip_prob = hourly_data['precipitation_probability'][i]
            output_line += f" | 🌧️  {precip_prob}%"

        if show_wind:
            wind_speed = hourly_data['wind_speed_10m'][i]
            output_line += f" | 💨 {wind_speed:.1f} km/h"

        print(output_line)


def get_weather_forecast(latitude, longitude, city_name, config):
    """
    Ottiene le previsioni meteorologiche per le coordinate specificate

    Args:
        latitude: Latitudine della città
        longitude: Longitudine della città
        city_name: Nome della città
        config: Tupla con configurazione (days, show_temp, show_wind, show_precip)
    """
    days, show_temp, show_wind, show_precip = config

    # API con parametri per temperatura, precipitazioni e velocità del vento
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,precipitation_probability,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
        f"&timezone=auto"
        f"&forecast_days={min(days, 7)}"  # API supporta max 7 giorni gratuiti
    )

    try:
        response = requests.get(weather_url, verify=False, timeout=10)
        response.raise_for_status()

        data = response.json()

        print(f"\n🌤️  PREVISIONI METEOROLOGICHE PER {city_name.upper()}")
        print("=" * 70)

        # Dati giornalieri
        daily = data['daily']
        print(f"\n📅 PREVISIONI GIORNALIERE ({days} giorni):")
        print("-" * 70)

        for i in range(min(days, len(daily['time']))):
            date = daily['time'][i]
            output_line = f"📆 {date}"

            if show_temp:
                temp_max = daily['temperature_2m_max'][i]
                temp_min = daily['temperature_2m_min'][i]
                output_line += f" | 🌡️  {temp_min:.1f}°C - {temp_max:.1f}°C"

            if show_precip:
                precip_sum = daily['precipitation_sum'][i]
                output_line += f" | 🌧️  {precip_sum:.1f}mm"

            if show_wind:
                wind_max = daily['wind_speed_10m_max'][i]
                output_line += f" | 💨 max {wind_max:.1f} km/h"

            print(output_line)

        # Se l'utente sceglie 1 giorno, mostra anche le previsioni orarie per oggi
        if days == 1:
            display_hourly_forecast(data['hourly'], show_temp, show_wind, show_precip)

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nella richiesta meteo: {e}")
        return False


def main():
    """Funzione principale per ottenere previsioni meteorologiche"""
    print("🌤️  PREVISIONI METEOROLOGICHE")
    print("=" * 40)

    # Richiedi input da utente (o usa valore predefinito per test in Docker)
    try:
        city_name = input("🏙️  Inserisci il nome della città: ").strip()
    except EOFError:
        # Se non c'è input (es. in Docker), usa valore predefinito
        city_name = "Roma"
        print(f"🏙️  Usando città predefinita: {city_name}")

    if not city_name:
        city_name = "Roma"
        print(f"🏙️  Usando città predefinita: {city_name}")

    # Ottieni coordinate della città
    latitude, longitude, found_city = get_city_coordinates(city_name)

    if latitude is None:
        print("❌ Impossibile ottenere le coordinate della città.")
        return

    # Ottieni preferenze utente
    user_config = get_user_preferences()

    # Ottieni previsioni meteorologiche
    success = get_weather_forecast(latitude, longitude, found_city, user_config)

    if success:
        print("\n✅ Previsioni ottenute con successo!")
    else:
        print("\n❌ Errore nell'ottenere le previsioni.")


def create_dockerfile():
    """Crea il file Dockerfile per la containerizzazione"""
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(DOCKERFILE_CONTENT)
        print("✅ Dockerfile creato.")


if __name__ == '__main__':
    create_dockerfile()
    main()
