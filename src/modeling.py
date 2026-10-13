import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

SEED = 42

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def make_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def build_pipeline(clf, smote=False):
    steps = [("pre", make_preprocessor())]
    if smote:  # applied only to the training part of each fit
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def candidate_models(pos_weight):
    models = {
        "Dummy (prior)": lambda: build_pipeline(DummyClassifier(strategy="prior")),
        "Logistic Regression (class weights)": lambda: build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5)),
        "Logistic Regression (SMOTE)": lambda: build_pipeline(
            LogisticRegression(max_iter=2000, C=0.5), smote=True),
        "Random Forest (class weights)": lambda: build_pipeline(
            RandomForestClassifier(n_estimators=200, min_samples_leaf=8, max_depth=10,
                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=1)),
    }
    if HAS_XGB:
        def xgb(**kw):
            return XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                                 n_jobs=1, **kw)
        models["XGBoost (scale_pos_weight)"] = lambda: build_pipeline(xgb(scale_pos_weight=pos_weight))
        models["XGBoost (SMOTE)"] = lambda: build_pipeline(xgb(), smote=True)
    return models


def cv_splitter():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def oof_proba(estimator, X, y):
    return cross_val_predict(estimator, X, y, cv=cv_splitter(), method="predict_proba", n_jobs=1)[:, 1]


def metrics_at(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "PR-AUC": average_precision_score(y, prob), "ROC-AUC": roc_auc_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def compare_models(X, y):
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    rows, oof = [], {}
    for name, make in candidate_models(pos_weight).items():
        prob = oof_proba(make(), X, y)
        oof[name] = prob
        rows.append({"Model": name, **metrics_at(y, prob)})
        print(f"  {name}: PR-AUC={rows[-1]['PR-AUC']:.4f}", flush=True)
    table = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return table, oof, candidate_models(pos_weight)


def calibrated(estimator, method):
    return CalibratedClassifierCV(estimator, method=method, cv=5)
