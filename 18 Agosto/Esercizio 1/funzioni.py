'''
Questo script contiene le funzioni per svolgere esercizio 1 del 18 Agosto.
'''

import pandas as pd
import numpy as np
from collections import Counter

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
    
def count_top5(df:pd.DataFrame) -> int:
    '''
    Questa funzione prende in input il dataframe derivante da lfile di testp e mi restituisce il conteggio totale delle parole presenti.
    
    Args:
        pd.DataFrame: DataFrame da leggere e analizzare.
        
    Returns:
        dizionario di output con parola:conteggio formato per la top5.
    '''

    words = []
    for col in df.columns:
        words.extend(df[col].astype(str).str.split().sum())
    
    count_words = Counter(words)
    top_5 = count_words.most_common(5)
    
    return {parola: conteggio for parola, conteggio in top_5}


    