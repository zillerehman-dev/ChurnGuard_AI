import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from src import business
from src.data import FEATURES, ROOT, load_xy
from src.explainability import ChurnExplainer
from src.modeling import SEED, calibrated, compare_models, metrics_at, oof_proba

MODELS, REPORTS = ROOT / "models", ROOT / "reports"
COSTS = dict(business.DEFAULT_COSTS)


def md_table(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def main():
    t0 = time.time()
    X, y, raw = load_xy()
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features; churn rate {y.mean():.3%}; "
          f"blank Total Charges: {X['Total Charges'].isna().sum()}; duplicates: {raw.duplicated().sum()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cltv_test = raw.loc[X_test.index, "CLTV"]  # reporting only

    print("Cross-validating candidates (5-fold stratified, train split only)...")
    comparison, oof, makers = compare_models(X_train, y_train)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.md").write_text(md_table(comparison.round(4)))
    best_name = comparison.loc[0, "Model"]
    make_best = makers[best_name]
    print(f"Selected by PR-AUC: {best_name}")

    # Calibration choice on out-of-fold probabilities (train only)
    cal_rows = [{"method": "none", "brier": brier_score_loss(y_train, oof[best_name]),
                 "log_loss": log_loss(y_train, oof[best_name])}]
    cal_oof = {"none": oof[best_name]}
    for method in ["sigmoid", "isotonic"]:
        p = oof_proba(calibrated(make_best(), method), X_train, y_train)
        cal_oof[method] = p
        cal_rows.append({"method": method, "brier": brier_score_loss(y_train, p), "log_loss": log_loss(y_train, p)})
    cal_table = pd.DataFrame(cal_rows)
    cal_table.to_csv(REPORTS / "calibration_comparison.csv", index=False)
    cal_method = cal_table.loc[cal_table.brier.idxmin(), "method"]
    print(cal_table.round(4).to_string(index=False), f"\nCalibration choice: {cal_method}")
    val_prob = cal_oof[cal_method]

    # Cost-based threshold on validation (out-of-fold) predictions, frozen before the test set
    threshold, curve = business.best_threshold(y_train, val_prob, COSTS)
    curve.to_csv(REPORTS / "threshold_curve.csv", index=False)
    val_policies = business.policy_comparison(y_train, val_prob, threshold, COSTS)
    val_policies.to_csv(REPORTS / "policy_comparison_validation.csv", index=False)
    print(f"Break-even probability {business.break_even(COSTS):.3f}; selected threshold {threshold:.2f}")

    # Final fit on training data, single evaluation on the untouched test set
    base = make_best().fit(X_train, y_train)
    final = (calibrated(make_best(), cal_method) if cal_method != "none" else make_best()).fit(X_train, y_train)
    test_prob = final.predict_proba(X_test)[:, 1]
    test_default = metrics_at(y_test, test_prob, 0.5)
    test_selected = metrics_at(y_test, test_prob, threshold)
    test_policies = business.impact_report(y_test, test_prob, threshold, COSTS, cltv_test)
    test_policies.to_csv(REPORTS / "policy_comparison_test.csv", index=False)
    print("Test PR-AUC {PR-AUC:.4f} ROC-AUC {ROC-AUC:.4f}".format(**test_selected))
    print(test_policies.to_string(index=False))

    # SHAP on the underlying (uncalibrated) pipeline
    background = X_train.sample(100, random_state=SEED)
    explainer = ChurnExplainer(base, background)
    shap_sample = X_train.sample(1000, random_state=SEED)
    explainer.summary_plot(shap_sample, REPORTS / "shap_summary.png")
    importance = explainer.global_importance(shap_sample)
    importance.rename("mean_abs_shap").to_csv(REPORTS / "shap_importance.csv")

    # Plots
    plt.figure(figsize=(6, 4))
    plt.plot(curve.threshold, curve.expected_cost)
    plt.axvline(threshold, color="red", ls="--", label=f"selected {threshold:.2f}")
    plt.axvline(0.5, color="gray", ls=":", label="0.50")
    plt.xlabel("Decision threshold"); plt.ylabel("Expected cost ($, train OOF)"); plt.legend()
    plt.title("Cost vs threshold"); plt.tight_layout(); plt.savefig(REPORTS / "threshold_cost.png", dpi=130); plt.close()

    plt.figure(figsize=(5, 5))
    for name, p in [("uncalibrated", oof[best_name]), (f"{cal_method}", val_prob)]:
        frac, mean = calibration_curve(y_train, p, n_bins=10, strategy="quantile")
        plt.plot(mean, frac, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed churn rate"); plt.legend()
    plt.title("Calibration (out-of-fold)"); plt.tight_layout(); plt.savefig(REPORTS / "calibration.png", dpi=130); plt.close()

    # Scored test table with reasons (used by the report and a quick sanity check)
    reasons = explainer.top_reasons(X_test.head(200))
    scored = business.score_customers(X_test.head(200), test_prob[:200], reasons, threshold, COSTS)
    scored.to_csv(REPORTS / "scored_test_sample.csv", index=False)

    # Artifacts
    joblib.dump(final, MODELS / "model.joblib", compress=3)
    joblib.dump(base, MODELS / "explainer_pipeline.joblib", compress=3)
    background.to_csv(MODELS / "background.csv", index=False)
    X_test.sample(5, random_state=1).to_csv(MODELS / "sample_input.csv", index=False)
    meta = {
        "model": best_name, "calibration": cal_method, "threshold": threshold,
        "features": FEATURES, "costs": COSTS, "break_even_probability": business.break_even(COSTS),
        "seed": SEED, "sklearn": sklearn.__version__,
        "cv_comparison": comparison.round(4).to_dict(orient="records"),
        "test_at_selected_threshold": {k: float(v) for k, v in test_selected.items()},
        "test_at_0.50": {k: float(v) for k, v in test_default.items()},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    }
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    summary = ["# Evaluation summary", "", "## CV comparison (train split, 5-fold OOF)", md_table(comparison.round(4)),
               "", "## Calibration (OOF)", md_table(cal_table.round(4)), f"Chosen: {cal_method}", "",
               f"## Threshold: {threshold:.2f} (break-even {business.break_even(COSTS):.3f})", "",
               "## Validation policies (train OOF)", md_table(val_policies), "",
               "## Test metrics", md_table(pd.DataFrame({"@selected": test_selected, "@0.50": test_default}).T.round(4).reset_index()),
               "", "## Test policy comparison (estimates under assumptions)", md_table(test_policies),
               "", test_policies.attrs.get("cltv_note", ""), "", "## Top SHAP features", md_table(importance.head(10).round(4).rename("mean_abs_shap").reset_index())]
    (REPORTS / "evaluation_summary.md").write_text("\n".join(summary))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
