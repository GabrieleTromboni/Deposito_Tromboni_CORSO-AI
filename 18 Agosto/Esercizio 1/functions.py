'''
Questo script contiene le funzioni per svolgere esercizio 1 del 18 Agosto.
'''

import pandas as pd
import numpy as np

def count_rows(df: pd.DataFrame) -> int:
    '''
    Questa funzione prende in input il dataframe derivante dal file di testo e mi restituisce il conteggio delle righe totali.
    
    Args:
        pd.DataFrame: DataFrame da leggere e analizzare.
        
    Returns:
        numero totale di righe del dataframe.
        '''
    
    return len(df)

def count_words(df:pd.DataFrame) -> int:
    '''
    Questa funzione prende in input il dataframe derivante da lfile di testp e mi restituisce il conteggio totale delle parole presenti.
    
    Args:
        pd.DataFrame: DataFrame da leggere e analizzare.
        
    Returns:
        numero totale di parole nel dataframe.
    '''

    total_words = 0
    for col in df.columns:
        total_words += df[col].astype(str).str.split().str.len().sum()
    return total_words
    
    