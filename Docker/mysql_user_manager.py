"""
Script per gestire utenti in un database MySQL containerizzato con Docker.
Permette di inserire e visualizzare utenti tramite interfaccia interattiva.
"""

import mysql.connector
import subprocess
import time
from mysql.connector import Error


class MySQLManager:
    """Classe per gestire la connessione e le operazioni MySQL"""
    
    def __init__(self, host="localhost", user="myuser", password="mypassword", 
                 database="mydb", port=3306):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Stabilisce la connessione al database MySQL"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port
            )
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                print("✅ Connessione al database MySQL riuscita!")
                return True
                
        except Error as e:
            print(f"❌ Errore di connessione al database: {e}")
            return False
    
    def create_users_table(self):
        """Crea la tabella users se non esiste"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nome VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                eta INT,
                data_creazione TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            print("✅ Tabella 'users' creata/verificata con successo!")
            
        except Error as e:
            print(f"❌ Errore nella creazione della tabella: {e}")
    
    def insert_user(self, nome, email, eta):
        """Inserisce un nuovo utente nel database"""
        try:
            insert_query = """
            INSERT INTO users (nome, email, eta) 
            VALUES (%s, %s, %s)
            """
            user_data = (nome, email, eta)
            
            self.cursor.execute(insert_query, user_data)
            self.connection.commit()
            
            print(f"✅ Utente '{nome}' inserito con successo!")
            return True
            
        except Error as e:
            print(f"❌ Errore nell'inserimento dell'utente: {e}")
            return False
    
    def get_all_users(self):
        """Recupera tutti gli utenti dal database"""
        try:
            select_query = "SELECT id, nome, email, eta, data_creazione FROM users"
            self.cursor.execute(select_query)
            users = self.cursor.fetchall()
            return users
            
        except Error as e:
            print(f"❌ Errore nel recupero degli utenti: {e}")
            return []
    
    def display_users(self):
        """Visualizza tutti gli utenti in formato tabellare"""
        users = self.get_all_users()
        
        if not users:
            print("📋 Nessun utente presente nel database.")
            return
        
        print("\n👥 UTENTI PRESENTI NEL DATABASE:")
        print("=" * 80)
        print(f"{'ID':<5} {'Nome':<20} {'Email':<30} {'Età':<5} {'Data Creazione':<20}")
        print("-" * 80)
        
        for user in users:
            user_id, nome, email, eta, data_creazione = user
            print(f"{user_id:<5} {nome:<20} {email:<30} {eta:<5} {str(data_creazione):<20}")
        
        print("-" * 80)
        print(f"Totale utenti: {len(users)}")
    
    def close_connection(self):
        """Chiude la connessione al database"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("🔌 Connessione al database chiusa.")


def check_docker_running():
    """Verifica se Docker è in esecuzione"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ Docker è disponibile.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker non è disponibile o non è in esecuzione.")
        return False


def create_mysql_container():
    """Crea e avvia il container MySQL"""
    print("\n🐳 Creazione container MySQL...")
    
    # Ferma e rimuove il container esistente se presente (pulizia completa)
    try:
        print("🧹 Pulizia container esistenti...")
        subprocess.run(['docker', 'stop', 'mysql-container'], 
                      capture_output=True, check=False)
        subprocess.run(['docker', 'rm', 'mysql-container'], 
                      capture_output=True, check=False)
        print("✅ Pulizia completata.")
    except Exception as e:
        print(f"⚠️ Errore durante la pulizia: {e}")
    
    # Crea ed esegue il nuovo container con --rm per auto-pulizia
    docker_command = [
        'docker', 'run', '-d', '--rm',  # Aggiunto --rm per auto-rimozione
        '--name', 'mysql-container',
        '-p', '3306:3306',
        '-e', 'MYSQL_ROOT_PASSWORD=rootpassword',
        '-e', 'MYSQL_DATABASE=mydb',
        '-e', 'MYSQL_USER=myuser',
        '-e', 'MYSQL_PASSWORD=mypassword',
        'mysql:latest'
    ]
    
    try:
        result = subprocess.run(docker_command, capture_output=True, text=True, check=True)
        print("✅ Container MySQL creato e avviato!")
        print("🔄 Container configurato per auto-rimozione quando fermato")
        print("⏳ Attendere qualche secondo per l'inizializzazione del database...")
        time.sleep(15)  # Attende che MySQL si inizializzi
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nella creazione del container: {e.stderr}")
        return False


def get_user_input():
    """Ottiene i dati dell'utente da input"""
    print("\n📝 INSERIMENTO NUOVO UTENTE")
    print("-" * 40)
    
    try:
        nome = input("Nome: ").strip()
        if not nome:
            print("⚠️ Il nome è obbligatorio!")
            return None
        
        email = input("Email: ").strip()
        if not email or "@" not in email:
            print("⚠️ Email non valida!")
            return None
        
        eta_str = input("Età: ").strip()
        try:
            eta = int(eta_str)
            if eta < 0 or eta > 150:
                print("⚠️ Età non valida!")
                return None
        except ValueError:
            print("⚠️ L'età deve essere un numero!")
            return None
        
        return nome, email, eta
        
    except KeyboardInterrupt:
        print("\n⚠️ Operazione annullata.")
        return None


def main_menu():
    """Menu principale dell'applicazione"""
    print("\n🏠 MENU PRINCIPALE")
    print("=" * 30)
    print("1️⃣  Inserire nuovo utente")
    print("2️⃣  Visualizzare tutti gli utenti")
    print("3️⃣  Ricreare container MySQL")
    print("4️⃣  Uscire")
    print("-" * 30)
    
    choice = input("Scegli un'opzione (1-4): ").strip()
    return choice


def main():
    """Funzione principale"""
    print("🗄️  GESTIONE UTENTI MYSQL CON DOCKER")
    print("=" * 50)
    
    # Verifica Docker
    if not check_docker_running():
        print("Installa e avvia Docker prima di continuare.")
        return
    
    # Crea il container MySQL
    if not create_mysql_container():
        print("Impossibile creare il container MySQL.")
        return
    
    # Inizializza il manager MySQL
    db_manager = MySQLManager()
    
    # Tenta la connessione
    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"\n🔄 Tentativo di connessione {attempt + 1}/{max_attempts}...")
        if db_manager.connect():
            break
        time.sleep(5)
    else:
        print("❌ Impossibile connettersi al database dopo 3 tentativi.")
        return
    
    # Crea la tabella users
    db_manager.create_users_table()
    
    # Menu principale
    try:
        while True:
            choice = main_menu()
            
            if choice == "1":
                # Inserire nuovo utente
                user_data = get_user_input()
                if user_data:
                    nome, email, eta = user_data
                    db_manager.insert_user(nome, email, eta)
            
            elif choice == "2":
                # Visualizzare utenti
                db_manager.display_users()
            
            elif choice == "3":
                # Ricreare container
                db_manager.close_connection()
                if create_mysql_container():
                    if db_manager.connect():
                        db_manager.create_users_table()
                        print("✅ Container ricreato e riconnesso!")
                    else:
                        print("❌ Errore nella riconnessione.")
                        break
            
            elif choice == "4":
                # Uscire
                print("👋 Uscita dal programma...")
                break
            
            else:
                print("⚠️ Scelta non valida. Riprova.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Programma interrotto dall'utente.")
    
    finally:
        # Chiudi connessione
        db_manager.close_connection()
        
        # Chiedi se fermare il container
        try:
            stop_container = input("\n🛑 Vuoi fermare il container MySQL? (s/n): ").strip().lower()
            if stop_container in ['s', 'si', 'y', 'yes']:
                print("🛑 Fermando container MySQL...")
                subprocess.run(['docker', 'stop', 'mysql-container'], 
                              capture_output=True, check=False)
                print("🛑 Container MySQL fermato e rimosso automaticamente.")
                
                # Chiedi se rimuovere anche l'immagine MySQL
                remove_image = input("🗑️ Vuoi rimuovere anche l'immagine MySQL? (s/n): ").strip().lower()
                if remove_image in ['s', 'si', 'y', 'yes']:
                    try:
                        subprocess.run(['docker', 'rmi', 'mysql:latest'], 
                                      capture_output=True, check=True)
                        print("🗑️ Immagine MySQL rimossa.")
                    except subprocess.CalledProcessError:
                        print("⚠️ Impossibile rimuovere l'immagine (potrebbe essere in uso).")
        except:
            pass


if __name__ == "__main__":
    main()
