import json
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src import business
from src.data import CATEGORICAL, FEATURES, NUMERIC, validate_input
from src.explainability import ChurnExplainer

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REQUIRED = ["model.joblib", "explainer_pipeline.joblib", "background.csv", "metadata.json"]
MAX_ROWS = 5000


class ArtifactError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_artifacts():
    missing = [f for f in REQUIRED if not (MODELS / f).exists()]
    if missing:
        raise ArtifactError("Missing model files: " + ", ".join(missing) + ". Run `python train.py` first.")
    meta = json.loads((MODELS / "metadata.json").read_text())
    if meta["features"] != FEATURES:
        raise ArtifactError("Saved model does not match the current feature schema. Re-run `python train.py`.")
    try:
        model = joblib.load(MODELS / "model.joblib")
        base = joblib.load(MODELS / "explainer_pipeline.joblib")
        explainer = ChurnExplainer(base, pd.read_csv(MODELS / "background.csv"))
    except Exception as exc:
        raise ArtifactError(f"Could not load model files ({exc}). Re-run `python train.py` in this environment.")
    ohe = base.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"]
    choices = {c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL, ohe.categories_)}
    return model, explainer, meta, choices


def predict_frame(X):
    """Single entry point for one-row and batch predictions."""
    model, explainer, meta, _ = load_artifacts()
    prob = model.predict_proba(X)[:, 1]
    reasons = explainer.top_reasons(X)
    return business.score_customers(X, prob, reasons, meta["threshold"], meta["costs"])


def example_csv_path():
    path = Path(tempfile.gettempdir()) / "churnguard_example_input.csv"
    sample = MODELS / "sample_input.csv"
    if sample.exists():
        pd.read_csv(sample)[FEATURES].to_csv(path, index=False)
    else:
        load_artifacts()
        pd.DataFrame(columns=FEATURES).to_csv(path, index=False)
    return str(path)


def predict_single(*values):
    try:
        row = dict(zip(FEATURES, values))
        if any(v is None or v == "" for k, v in row.items() if k != "Total Charges"):
            return "Please fill in every field."
        if row["Monthly Charges"] < 0 or (row["Total Charges"] or 0) < 0 or row["Tenure Months"] < 0:
            return "Tenure and charges cannot be negative."
        X = validate_input(pd.DataFrame([row]))
        res = predict_frame(X).iloc[0]
        _, _, meta, _ = load_artifacts()
        prob = res["Churn Probability"]
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[res["Risk Category"]]
        reasons = "\n".join(f"{i}. {res[f'Reason {i}']}" for i in (1, 2, 3))
        verdict = ("Offering a retention incentive has a positive expected net benefit."
                   if res["Recommended Action"] == "Offer retention"
                   else "The churn probability is below the break-even point, so an offer is not expected to pay off.")
        return (f"### {icon} {res['Risk Category']} risk: {prob:.1%} churn probability\n"
                f"**Recommendation:** {res['Recommended Action']}  \n"
                f"**Expected net benefit of an offer:** ${res['Expected Net Benefit ($)']:,.2f}\n\n"
                f"**Top drivers**\n{reasons}\n\n"
                f"{verdict} Decision threshold {meta['threshold']:.0%}; "
                f"break-even probability {meta['break_even_probability']:.1%}. "
                f"Financial figures are estimates under illustrative assumptions "
                f"(loss ${meta['costs']['loss_cost']:.0f}, offer ${meta['costs']['offer_cost']:.0f}, "
                f"{meta['costs']['offer_success']:.0%} success).")
    except ArtifactError as exc:
        return f"⚠️ {exc}"
    except ValueError as exc:
        return f"⚠️ {exc}"
    except Exception as exc:
        return f"⚠️ Unexpected error while predicting: {exc}"


def predict_batch(file):
    empty = pd.DataFrame()
    if file is None:
        return "Upload a CSV file first.", empty, None
    try:
        df = pd.read_csv(file if isinstance(file, str) else file.name)
        if len(df) > MAX_ROWS:
            return f"Please upload at most {MAX_ROWS:,} rows (received {len(df):,}).", empty, None
        X = validate_input(df)  # drops target/leakage columns by selecting model features only
        result = predict_frame(X)
        if "CustomerID" in df.columns:
            result.insert(0, "CustomerID", df["CustomerID"].values)
        out = Path(tempfile.gettempdir()) / "churnguard_predictions.csv"
        result.to_csv(out, index=False)
        high = int((result["Risk Category"] == "High").sum())
        msg = (f"Scored {len(result):,} customers. {high:,} high risk; "
               f"{int((result['Recommended Action'] == 'Offer retention').sum()):,} recommended for an offer.")
        return msg, result.head(500), str(out)
    except pd.errors.EmptyDataError:
        return "The uploaded file is empty.", empty, None
    except (ArtifactError, ValueError) as exc:
        return f"⚠️ {exc}", empty, None
    except Exception as exc:
        return f"⚠️ Could not process the file: {exc}", empty, None


def build_ui():
    try:
        _, _, meta, choices = load_artifacts()
        status = (f"Model: {meta['model']} · calibration: {meta['calibration']} · "
                  f"threshold {meta['threshold']:.0%}")
    except ArtifactError as exc:
        meta, choices, status = None, {c: [] for c in CATEGORICAL}, f"⚠️ {exc}"

    defaults = {"Gender": "Female", "Contract": "Month-to-month", "Internet Service": "Fiber optic",
                "Payment Method": "Electronic check", "Senior Citizen": "No", "Phone Service": "Yes"}

    with gr.Blocks(title="ChurnGuard") as demo:
        gr.Markdown("# ChurnGuard\nCustomer churn prediction and retention decisions for a telecom provider.")
        gr.Markdown(status)
        with gr.Tab("Predict one customer"):
            inputs = {}
            with gr.Row():
                with gr.Column():
                    inputs["Tenure Months"] = gr.Slider(0, 72, value=6, step=1, label="Tenure Months")
                    inputs["Monthly Charges"] = gr.Number(value=75.0, minimum=0, maximum=500, label="Monthly Charges ($)")
                    inputs["Total Charges"] = gr.Number(value=450.0, minimum=0, label="Total Charges ($)")
                    for c in ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines"]:
                        opts = choices[c]
                        inputs[c] = gr.Dropdown(opts, value=defaults.get(c, opts[0] if opts else None), label=c)
                with gr.Column():
                    for c in CATEGORICAL[6:]:
                        opts = choices[c]
                        inputs[c] = gr.Dropdown(opts, value=defaults.get(c, opts[0] if opts else None), label=c)
            btn = gr.Button("Predict", variant="primary")
            result = gr.Markdown()
            btn.click(predict_single, [inputs[f] for f in FEATURES], result)

        with gr.Tab("Upload a CSV"):
            gr.Markdown("Upload a CSV with these columns (extra columns such as CustomerID or churn labels are ignored):\n\n"
                        + ", ".join(f"`{f}`" for f in FEATURES))
            example = gr.DownloadButton("Download example input CSV", value=example_csv_path() if meta else None)
            upload = gr.File(label="Customer CSV", file_types=[".csv"])
            run = gr.Button("Score customers", variant="primary")
            msg = gr.Markdown()
            table = gr.Dataframe(label="Predictions (first 500 rows)")
            download = gr.File(label="Download full results")
            run.click(predict_batch, upload, [msg, table, download])
        gr.Markdown("Financial figures are estimates based on configurable assumptions, not proven savings.")
    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
