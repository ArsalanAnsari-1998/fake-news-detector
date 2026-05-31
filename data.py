import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/WELFake_Dataset.csv")

# Combine title + text for richer features
df["statement"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["binary"] = df["label"].astype(int)

df = df[["statement", "binary"]].dropna()

train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["binary"]
)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv",  index=False)

print(f"Train: {len(train)} | Test: {len(test)}")
print("Label split:\n", train["binary"].value_counts())