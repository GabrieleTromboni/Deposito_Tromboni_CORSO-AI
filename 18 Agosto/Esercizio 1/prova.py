'''
Questo script svolge le richieste dell'esercizio 1 del 18 Agosto:

- Leggere un file 'input.txt' (UTF-8)
- Analizzare il file
- Stampare numero totale di righe
- Stampare numero totale di parole
- Dare in output una top5 delle parole più frequenti nel formato parola:conteggio
'''

# IMPORT LIBRARIES
import pandas as pd
import numpy as np

# IMPORT SCRIPT
from funzioni import count_rows, count_words, count_top5

def analisi_file():

    # LETTURA FILE 'input.txt'
    dati = pd.read_csv('input.txt', encoding='utf-8', delimiter='\t')

    # CONTEGGIO RIGHE DEL DATAFRAME
    total_rows = count_rows(dati)
    print(f"Il conteggio totale delle righe è: {total_rows}")
    
    # CONTEGGIO PAROLE TOTALI DEL DATAFRAME
    total_words = count_words(dati)
    print(f"Il conteggio totale delle parole è: {total_words}")

    # OUTPUT TOP-5 PAROLE:CONTEGGIO
    top_parole = count_top5(dati)
    print("Le top 5 parole sono:")
    for parola, conteggio in top_parole.items():
        print(f"{parola}: {conteggio}")

if __name__ == 'main':
    analisi_file()