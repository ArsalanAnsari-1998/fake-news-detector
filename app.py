import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

st.title("🔍 Fake News Detector")
st.caption("TF-IDF + Logistic Regression · Explained with SHAP")

@st.cache_resource
def load_model():
    pipeline  = joblib.load("model.pkl")
    tfidf     = pipeline.named_steps["tfidf"]
    clf       = pipeline.named_steps["clf"]
    explainer = shap.LinearExplainer(clf, shap.maskers.Independent(
        tfidf.transform(pd.read_csv("train.csv")["statement"].dropna().iloc[:100])
    ))
    return tfidf, clf, explainer

tfidf, clf, explainer = load_model()

st.markdown("### Paste a news headline or article")
text = st.text_area("", placeholder="e.g. Scientists discover water on Mars...", height=150, label_visibility="collapsed")

if st.button("Analyse", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        X = tfidf.transform([text])

        pred  = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        confidence = proba[pred] * 100

        if pred == 1:
            st.success(f"✅ REAL NEWS — {confidence:.1f}% confidence")
        else:
            st.error(f"🚨 FAKE NEWS — {confidence:.1f}% confidence")

        st.markdown("---")
        st.markdown("#### Why this verdict? (SHAP explanation)")
        st.caption("Bars pushed right → words pushing toward REAL. Bars pushed left → words pushing toward FAKE.")

        sv = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values        = sv[0],
                base_values   = explainer.expected_value,
                feature_names = tfidf.get_feature_names_out()
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("#### Top contributing words")
        feature_names = tfidf.get_feature_names_out()
        shap_vals     = sv[0]

        top_idx  = abs(shap_vals).argsort()[-10:][::-1]
        top_df   = pd.DataFrame({
            "Word"   : feature_names[top_idx],
            "Impact" : shap_vals[top_idx].round(4),
            "Direction": ["→ Real" if v > 0 else "→ Fake" for v in shap_vals[top_idx]]
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)