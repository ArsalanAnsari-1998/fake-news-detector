import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def load_and_clean(path):
    df = pd.read_csv(path)
    return df[["statement", "binary"]].dropna()

test = load_and_clean("test.csv")

pipeline  = joblib.load("model.pkl")
tfidf     = pipeline.named_steps["tfidf"]
clf       = pipeline.named_steps["clf"]

print("Transforming test data...")
X_test_tfidf = tfidf.transform(test["statement"])

print("Building SHAP explainer...")
explainer   = shap.LinearExplainer(clf, X_test_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

# --- Plot 1: Global feature importance (top 20 words) ---
print("Saving global SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_test_tfidf,
    feature_names=tfidf.get_feature_names_out(),
    plot_type="bar",
    max_display=20,
    show=False
)
plt.title("Top 20 words driving Fake vs Real predictions")
plt.tight_layout()
plt.savefig("shap_global.png", dpi=150)
plt.clf()
print("Saved: shap_global.png")

# --- Plot 2: Local explanation for one fake article ---
fake_idx    = test[test["binary"] == 0].index[0]
sample_text = test.loc[fake_idx, "statement"]
X_sample    = tfidf.transform([sample_text])
sv_sample   = explainer.shap_values(X_sample)

print("\nSample article (first 300 chars):")
print(sample_text[:300])

shap.waterfall_plot(
    shap.Explanation(
        values         = sv_sample[0],
        base_values    = explainer.expected_value,
        feature_names  = tfidf.get_feature_names_out()
    ),
    max_display=15,
    show=False
)
plt.title("Why this article was flagged as FAKE")
plt.tight_layout()
plt.savefig("shap_local.png", dpi=150)
plt.clf()
print("Saved: shap_local.png")