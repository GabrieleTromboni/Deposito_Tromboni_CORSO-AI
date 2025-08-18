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
from functions.py import count_rows

def main(args):

    # LETTURA FILE 'input.txt'
    dati = pd.read_csv('input.txt', encoding='utf-8', delimiter='\t')

    # CONTEGGIO RIGHE DEL DATAFRAME
    total_rows = count_rows(dati)

    # CONTEGGIO PAROLE TOTALI DEL DATAFRAME

    # OUTPUT TOP-5 PAROLE:CONTEGGIO




