import pandas as pd
import numpy as np

def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df
