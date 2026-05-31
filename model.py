import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def load_and_clean(path):
    df = pd.read_csv(path)
    return df[["statement", "binary"]].dropna()

train = load_and_clean("train.csv")
test  = load_and_clean("test.csv")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf",   LogisticRegression(max_iter=1000))
])

print("Training model...")
pipeline.fit(train["statement"], train["binary"])

preds = pipeline.predict(test["statement"])
print(classification_report(test["binary"], preds, target_names=["Fake", "Real"]))

joblib.dump(pipeline, "model.pkl")
print("Model saved to model.pkl")