import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import business
from src.data import CATEGORICAL, FEATURES, validate_input
from src.explainability import ChurnExplainer

MODELS = Path(__file__).resolve().parent / "models"
REQUIRED = ["model.joblib", "explainer_pipeline.joblib", "background.csv", "metadata.json"]
MAX_ROWS = 5000

st.set_page_config(page_title="ChurnGuard", layout="wide")


@st.cache_resource
def load_artifacts():
    missing = [f for f in REQUIRED if not (MODELS / f).exists()]
    if missing:
        raise RuntimeError("Missing model files: " + ", ".join(missing) + ". Run `python train.py` first.")
    meta = json.loads((MODELS / "metadata.json").read_text())
    if meta["features"] != FEATURES:
        raise RuntimeError("Saved model does not match the feature schema. Re-run `python train.py`.")
    model = joblib.load(MODELS / "model.joblib")
    base = joblib.load(MODELS / "explainer_pipeline.joblib")
    explainer = ChurnExplainer(base, pd.read_csv(MODELS / "background.csv"))
    ohe = base.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    choices = {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL, ohe.categories_)}
    return model, explainer, meta, choices


def predict_frame(X):
    model, explainer, meta, _ = load_artifacts()
    prob = model.predict_proba(X)[:, 1]
    return business.score_customers(X, prob, explainer.top_reasons(X), meta["threshold"], meta["costs"])


st.title("ChurnGuard")
st.caption("Customer churn prediction and retention decisions for a telecom provider.")

try:
    _, _, meta, choices = load_artifacts()
except Exception as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

st.caption(f"Model: {meta['model']} · calibration: {meta['calibration']} · threshold {meta['threshold']:.0%}")
tab_one, tab_batch = st.tabs(["Predict one customer", "Upload a CSV"])

with tab_one:
    with st.form("single"):
        left, right = st.columns(2)
        row = {}
        with left:
            row["Tenure Months"] = st.slider("Tenure Months", 0, 72, 6)
            row["Monthly Charges"] = st.number_input("Monthly Charges ($)", 0.0, 500.0, 75.0)
            row["Total Charges"] = st.number_input("Total Charges ($)", 0.0, 20000.0, 450.0)
            for c in ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines"]:
                row[c] = st.selectbox(c, choices[c])
        with right:
            for c in CATEGORICAL[6:]:
                default = choices[c].index("Month-to-month") if c == "Contract" else 0
                row[c] = st.selectbox(c, choices[c], index=default)
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        try:
            res = predict_frame(validate_input(pd.DataFrame([row]))).iloc[0]
            icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[res["Risk Category"]]
            st.subheader(f"{icon} {res['Risk Category']} risk: {res['Churn Probability']:.1%} churn probability")
            st.markdown(f"**Recommendation:** {res['Recommended Action']}  \n"
                        f"**Expected net benefit of an offer:** ${res['Expected Net Benefit ($)']:,.2f}")
            st.markdown("**Top drivers**")
            for i in (1, 2, 3):
                st.markdown(f"{i}. {res[f'Reason {i}']}")
            c = meta["costs"]
            st.caption(f"Threshold {meta['threshold']:.0%}; break-even {meta['break_even_probability']:.1%}. "
                       f"Estimates under illustrative assumptions (loss ${c['loss_cost']:.0f}, "
                       f"offer ${c['offer_cost']:.0f}, {c['offer_success']:.0%} success).")
        except ValueError as exc:
            st.error(f"⚠️ {exc}")
        except Exception as exc:
            st.error(f"⚠️ Unexpected error while predicting: {exc}")

with tab_batch:
    st.markdown("Required columns (extra columns such as CustomerID or churn labels are ignored):")
    st.code(", ".join(FEATURES), language=None)
    sample = MODELS / "sample_input.csv"
    if sample.exists():
        st.download_button("Download example input CSV", pd.read_csv(sample)[FEATURES].to_csv(index=False),
                           "churnguard_example_input.csv", "text/csv")
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            if len(df) > MAX_ROWS:
                raise ValueError(f"Please upload at most {MAX_ROWS:,} rows (received {len(df):,}).")
            result = predict_frame(validate_input(df))
            if "CustomerID" in df.columns:
                result.insert(0, "CustomerID", df["CustomerID"].values)
            offers = int((result["Recommended Action"] == "Offer retention").sum())
            st.success(f"Scored {len(result):,} customers; {offers:,} recommended for an offer.")
            st.dataframe(result.head(500), use_container_width=True)
            st.download_button("Download full results", result.to_csv(index=False),
                               "churnguard_predictions.csv", "text/csv")
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
        except ValueError as exc:
            st.error(f"⚠️ {exc}")
        except Exception as exc:
            st.error(f"⚠️ Could not process the file: {exc}")

st.caption("Financial figures are estimates based on configurable assumptions, not proven savings.")
