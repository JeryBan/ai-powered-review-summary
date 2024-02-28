
import pandas as pd

def clean_dataframe(df: pd.core.frame.DataFrame):
    '''Applies some basing preprocess of dataframe'''
    
    # remove rows from column Review if they are not of type string
    df = df.drop(df[df['Review'].apply(lambda x: not isinstance(x, str))].index)
    
    # remove rows that contain only special symbols and not words
    df = df[~df['Review'].str.contains(r'^[\W_]+$')]

    # turn Rating column to numeric values and drop the rows that could not be converted
    df.loc[:, 'Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df.dropna()

    return df
