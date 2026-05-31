import pandas as pd

def load_and_clean(path):
    df = pd.read_csv(path)
    return df[["statement", "binary"]].dropna()

train = load_and_clean("train.csv")
test  = load_and_clean("test.csv")