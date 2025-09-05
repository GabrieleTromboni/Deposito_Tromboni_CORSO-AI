
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
