import pandas as pd

def encode_categorical(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.Categorical(df[column]).codes
    return df