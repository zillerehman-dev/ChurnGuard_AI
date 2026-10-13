import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from .data import CATEGORICAL, NUMERIC


def _positive_class(values):
    """Normalise SHAP output to (n_samples, n_features) for the churn class."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


class ChurnExplainer:
    """SHAP explanations from a fitted pipeline (steps 'pre' and 'clf'), aggregated to original features."""

    def __init__(self, pipeline, background):
        self.pre = pipeline.named_steps["pre"]
        clf = pipeline.named_steps["clf"]
        bg = self.pre.transform(background)
        if isinstance(clf, LogisticRegression):
            self.explainer = shap.LinearExplainer(clf, bg)
        else:
            self.explainer = shap.TreeExplainer(clf)
        self.groups = self._groups()

    def _groups(self):
        ohe = self.pre.named_transformers_["cat"].named_steps["onehot"]
        owner = list(NUMERIC)
        for col, cats in zip(CATEGORICAL, ohe.categories_):
            owner += [col] * len(cats)
        return np.array(owner)

    def transformed_shap(self, X):
        data = self.pre.transform(X)
        return data, _positive_class(self.explainer.shap_values(data))

    def feature_shap(self, X):
        """Per-original-feature SHAP values (one-hot columns summed)."""
        _, sv = self.transformed_shap(X)
        cols = NUMERIC + CATEGORICAL
        return pd.DataFrame({c: sv[:, self.groups == c].sum(axis=1) for c in cols}, index=range(len(X)))

    def top_reasons(self, X, k=3):
        shap_df = self.feature_shap(X)
        reasons = []
        for i in range(len(X)):
            row = X.iloc[i]
            top = shap_df.iloc[i].abs().sort_values(ascending=False).index[:k]
            reasons.append([_label(c, row[c], shap_df.iloc[i][c]) for c in top])
        return reasons

    def summary_plot(self, X, path):
        data, sv = self.transformed_shap(X)
        names = [n.split("__", 1)[1] for n in self.pre.get_feature_names_out()]
        plt.figure()
        shap.summary_plot(sv, data, feature_names=names, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()

    def global_importance(self, X):
        return self.feature_shap(X).abs().mean().sort_values(ascending=False)


def _label(col, value, contribution):
    direction = "raises churn risk" if contribution > 0 else "lowers churn risk"
    if col in NUMERIC:
        shown = f"{value:,.0f}" if col == "Tenure Months" else f"${value:,.2f}" if pd.notna(value) else "n/a"
        shown = f"{shown} months" if col == "Tenure Months" else shown
    else:
        shown = str(value)
    return f"{col}: {shown} ({direction})"
