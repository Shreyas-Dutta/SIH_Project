"""Train and evaluate heatwave forecasting models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from ..config import BASE_DIR
from ..features.engine import chronological_split, prepare_model_features, log_feature_summary
from ..utils.helpers import log


MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])
    return pre


def _metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    result = {
        "accuracy": float(np.mean(pred == y_true)),
        "roc_auc": None if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, prob)),
        "pr_auc": None if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "positive": int(np.sum(y_true == 1)),
        "negative": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(y_true, pred, zero_division=0),
    }
    return result


def _best_threshold(y_true, prob, minimum_recall=0.75):
    """Choose threshold to minimize both error types on VALIDATION only.

    Primary objective is balanced accuracy (equivalent to minimizing the mean
    false-positive rate and false-negative rate), with a recall floor so the
    alert system does not become overly conservative. F1 and precision break ties.
    """
    candidates = []
    for t in np.linspace(0.001, 0.999, 999):
        pred = (prob >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, pred)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        candidates.append({
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(bal_acc),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
        })

    constrained = [c for c in candidates if c["recall"] >= minimum_recall]
    pool = constrained if constrained else candidates
    best = max(
        pool,
        key=lambda c: (
            c["balanced_accuracy"],
            c["f1"],
            c["precision"],
            c["recall"],
        ),
    )
    return best["threshold"], best["f1"], best, candidates


def _save_diagnostics(model, X_train, y_train, y_test, test_prob, threshold, target):
    pred=(test_prob>=threshold).astype(int)
    cm=confusion_matrix(y_test,pred,labels=[0,1])
    pd.DataFrame(cm,index=["actual_0","actual_1"],columns=["predicted_0","predicted_1"]).to_csv(MODEL_DIR/f"confusion_matrix_{target}.csv")
    # Feature importance at original-column level when supported.
    rows=[]; est=model.named_steps["model"]; pre=model.named_steps["preprocess"]
    try:
        names=pre.get_feature_names_out(); vals=None
        if hasattr(est,"coef_"): vals=np.abs(np.ravel(est.coef_))
        elif hasattr(est,"feature_importances_"): vals=np.ravel(est.feature_importances_)
        if vals is not None and len(vals)==len(names):
            rows=sorted(zip(names,vals),key=lambda x:x[1],reverse=True)
    except Exception: pass
    pd.DataFrame(rows,columns=["feature","importance"]).head(100).to_csv(MODEL_DIR/f"feature_importance_{target}.csv",index=False)


def _balance_training_data(train: pd.DataFrame, target: str, negative_to_positive: int = 3) -> tuple[pd.DataFrame, dict]:
    """Keep the real training prevalence and use model class weights instead.

    The previous version undersampled negatives and then applied additional class
    weighting. That combination distorted probabilities and made the saved threshold
    inconsistent with live prediction. Keeping all training rows gives the learner
    more information and lets the classifier handle imbalance directly.
    """
    t = train.copy()
    pos = int((t[target] == 1).sum())
    neg = int((t[target] == 0).sum())
    return t, {
        "method": "full_training_set_class_weight",
        "negative_to_positive": negative_to_positive,
        "original_rows": int(len(train)),
        "original_positive": pos,
        "original_negative": neg,
        "balanced_rows": int(len(t)),
        "balanced_positive": pos,
        "balanced_negative": neg,
        "original_positive_rate": float(pos / max(1, len(t))),
        "balanced_positive_rate": float(pos / max(1, len(t))),
    }


def _fit_train_climatology(train: pd.DataFrame):
    """Fit district-month climatology using TRAIN ONLY."""
    t = train.copy()
    t["month"] = pd.to_datetime(t["date"]).dt.month
    keys = ["district", "month"]
    specs = [
        ("temperature_max_c", "district_tmax_p90_c", "district_tmax_p95_c"),
        ("wbgt_estimated_c", "district_wbgt_p90_c", "district_wbgt_p95_c"),
        ("heat_index_c", "district_hi_p90_c", "district_hi_p95_c"),
        ("temperature_min_c", "district_tmin_p90_c", "district_tmin_p95_c"),
    ]
    cl = t[keys].drop_duplicates().copy()
    for value, p90, p95 in specs:
        if value in t.columns:
            g = t.groupby(keys)[value]
            stats = pd.DataFrame({
                p90: g.quantile(0.90),
                p95: g.quantile(0.95),
            }).reset_index()
            cl = cl.merge(stats, on=keys, how="left")
    normal = t.groupby(keys).agg(
        tmax_normal_c=("temperature_max_c", "median"),
        wbgt_normal_c=("wbgt_estimated_c", "median"),
    ).reset_index()
    cl = cl.merge(normal, on=keys, how="left")
    return cl


def _apply_train_climatology(part: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    """Apply TRAIN-fitted climatology to any partition."""
    d = part.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.month
    d = d.drop(columns=[c for c in [
        "district_tmax_p90_c", "district_tmax_p95_c",
        "district_wbgt_p90_c", "district_wbgt_p95_c",
        "district_hi_p90_c", "district_hi_p95_c",
        "district_tmin_p90_c", "district_tmin_p95_c",
        "temperature_monthly_normal_c", "wbgt_monthly_normal_c",
    ] if c in d.columns], errors="ignore")
    d = d.merge(cl, on=["district", "month"], how="left", suffixes=("", "_fit"))
    d["temperature_monthly_normal_c"] = d["tmax_normal_c"]
    d["temperature_anomaly_c"] = d["temperature_max_c"] - d["tmax_normal_c"]
    d["wbgt_monthly_normal_c"] = d["wbgt_normal_c"]
    d["wbgt_anomaly_c"] = d["wbgt_estimated_c"] - d["wbgt_normal_c"]
    d["tmax_above_p90"] = (d["temperature_max_c"] > d["district_tmax_p90_c"]).astype("int8")
    d["tmax_above_p95"] = (d["temperature_max_c"] > d["district_tmax_p95_c"]).astype("int8")
    d["wbgt_above_p90"] = (d["wbgt_estimated_c"] > d["district_wbgt_p90_c"]).astype("int8")
    d["wbgt_above_p95"] = (d["wbgt_estimated_c"] > d["district_wbgt_p95_c"]).astype("int8")
    d["heat_index_above_p90"] = (d["heat_index_c"] > d["district_hi_p90_c"]).astype("int8")
    d["heat_index_above_p95"] = (d["heat_index_c"] > d["district_hi_p95_c"]).astype("int8")
    d["hot_night"] = (d["temperature_min_c"] > d["district_tmin_p90_c"]).astype("int8")
    d = d.sort_values(["district", "date"]).reset_index(drop=True)
    g = d.groupby("district", group_keys=False)
    def streak(s):
        groups = s.eq(0).cumsum()
        return (s.groupby(groups).cumcount() + 1).where(s.eq(1), 0)
    d["heatwave_day_flag"] = ((d["tmax_above_p95"] == 1) | (d["wbgt_above_p95"] == 1)).astype("int8")
    d["hot_day_streak"] = g["heatwave_day_flag"].transform(streak)
    d["extreme_wbgt_streak"] = g["wbgt_above_p95"].transform(streak)
    d["hot_night_3d_count"] = g["hot_night"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    return d


def _rebuild_target(part: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rebuild heatwave labels from future Tmax/WBGT and TRAIN-fitted climo."""
    d = part.copy().sort_values(["district", "date"]).reset_index(drop=True)
    g = d.groupby("district", group_keys=False)
    flags = []
    # The exported ML CSV may not retain future Tmax columns. Recreate them from
    # the chronological weather series before constructing the target.
    for h in range(1, 4):
        tcol = f"future_tmax_t_plus_{h}"
        wcol = f"future_wbgt_t_plus_{h}"
        if tcol not in d.columns and "temperature_max_c" in d.columns:
            d[tcol] = g["temperature_max_c"].shift(-h)
        if wcol not in d.columns and "wbgt_estimated_c" in d.columns:
            d[wcol] = g["wbgt_estimated_c"].shift(-h)
        if tcol not in d.columns or wcol not in d.columns:
            d[target] = np.nan
            return d
        flags.append(((d[tcol] > d["district_tmax_p95_c"]) | (d[wcol] > d["district_wbgt_p95_c"])).astype(float))
    f = pd.concat(flags, axis=1)
    d[target] = (f.sum(axis=1) >= 2).astype(float)
    d.loc[f.isna().any(axis=1), target] = np.nan
    return d

def _prior_correct_probability(prob, sampled_positive_rate: float, true_positive_rate: float):
    """Undo probability shift caused by negative undersampling (prior correction)."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    ps = np.clip(float(sampled_positive_rate), 1e-6, 1 - 1e-6)
    pt = np.clip(float(true_positive_rate), 1e-6, 1 - 1e-6)
    sampled_odds = p / (1 - p)
    true_prior_odds = pt / (1 - pt)
    sampled_prior_odds = ps / (1 - ps)
    corrected_odds = sampled_odds * (true_prior_odds / sampled_prior_odds)
    return corrected_odds / (1 + corrected_odds)

def train_models(dataset: pd.DataFrame, target="heatwave_next_3d"):
    work, X, y, feature_columns = prepare_model_features(dataset, target=target)
    log_feature_summary(work, feature_columns, target)

    # Strict temporal leakage control: fit district-month climatology on TRAIN ONLY
    # and apply the same frozen climatology to validation and test. Rebuild the
    # target with those frozen thresholds as well.
    train, valid, test = chronological_split(work)
    train_climatology = _fit_train_climatology(train)
    train = _apply_train_climatology(train, train_climatology)
    valid = _apply_train_climatology(valid, train_climatology)
    test = _apply_train_climatology(test, train_climatology)
    train = _rebuild_target(train, target).dropna(subset=[target]).copy()
    valid = _rebuild_target(valid, target).dropna(subset=[target]).copy()
    test = _rebuild_target(test, target).dropna(subset=[target]).copy()

    # Keep every training observation. The learner handles imbalance directly;
    # this avoids the probability distortion caused by undersampling + reweighting.
    balanced_train, balance_stats = _balance_training_data(train, target, negative_to_positive=3)
    X_train = balanced_train[feature_columns]
    y_train = balanced_train[target].astype(int)
    X_valid = valid[feature_columns]
    y_valid = valid[target].astype(int)
    X_test = test[feature_columns]
    y_test = test[target].astype(int)

    log(f"[BALANCE] Original train: {balance_stats.get('original_positive', 0):,} positive / {balance_stats.get('original_negative', 0):,} negative")
    log(f"[BALANCE] Training rows retained: {balance_stats.get('balanced_rows', 0):,} ({balance_stats.get('balanced_positive_rate', 0)*100:.2f}% positive)")
    log(f"[BALANCE] Class imbalance handled by model class weights; validation/test remain untouched")

    if y_train.nunique() < 2:
        raise RuntimeError("Training data contains only one target class. Cannot train classifier.")
    if y_test.nunique() < 2:
        log("[WARNING] Test set contains one class; ROC-AUC/PR-AUC may be unavailable.")

    pre = _build_preprocessor(X_train)

    # Primary production model: gradient-boosted trees capture nonlinear
    # heat/humidity interactions while class weighting protects the minority event.
    if XGBClassifier is not None:
        # Tuned conservative class weight. Full prevalence is retained; a very
        # large ratio over-alerts and increases false positives.
        pos = 7.0
        models = {
            "xgboost": XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.04,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=2.0,
                gamma=0.05,
                eval_metric="logloss",
                scale_pos_weight=pos,
                random_state=42,
                n_jobs=-1,
            )
        }
    else:
        models = {
            "logistic_regression": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            )
        }

    results = {}
    fitted = {}

    for name, estimator in models.items():
        log("")
        log(f"[MODEL] Training {name}...")

        pipe = Pipeline([
            ("preprocess", pre),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)

        valid_prob_raw = pipe.predict_proba(X_valid)[:, 1]
        test_prob_raw = pipe.predict_proba(X_test)[:, 1]

        # No probability-prior correction is needed because the full training
        # prevalence is retained. Threshold is selected directly on validation.
        valid_prob = valid_prob_raw
        test_prob = test_prob_raw

        threshold, best_valid_f1, threshold_selection, threshold_curve = _best_threshold(
            y_valid.to_numpy(), valid_prob, minimum_recall=0.75
        )
        valid_metrics = _metrics(y_valid.to_numpy(), valid_prob, threshold)
        valid_metrics["threshold_selection"] = threshold_selection
        valid_metrics["threshold_recall_constraint"] = 0.75
        valid_metrics["threshold_constraint_satisfied"] = bool(
            threshold_selection["recall"] >= 0.75
        )
        test_metrics = _metrics(y_test.to_numpy(), test_prob, threshold)

        results[name] = {
            "validation": valid_metrics,
            "threshold_search": threshold_curve,
            "test": test_metrics,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "balance": balance_stats,
            "true_training_prevalence": float(train[target].astype(int).mean()),
            "balanced_training_prevalence": float(y_train.mean()),
            "climatology_fit": "train_only",
        }

        fitted[name] = pipe

        log(f"  Auto threshold:  {threshold:.2f} (validation only)")
        log(f"  Threshold recall: {valid_metrics['recall']:.3f}")
        log(f"  Validation F1:    {valid_metrics['f1']:.3f}")
        log(f"  Test F1:       {test_metrics['f1']:.3f}")
        if test_metrics["roc_auc"] is not None:
            log(f"  Test ROC-AUC:  {test_metrics['roc_auc']:.3f}")
        if test_metrics["pr_auc"] is not None:
            log(f"  Test PR-AUC:   {test_metrics['pr_auc']:.3f}")

    # Choose the model using validation PR-AUC, falling back to F1.
    def score(item):
        metrics = item[1]["validation"]
        return metrics["pr_auc"] if metrics["pr_auc"] is not None else metrics["f1"]

    best_name, best_result = max(results.items(), key=score)
    best_model = fitted[best_name]

    # Keep a single live model artifact used by the forecast layer.
    compact_path = MODEL_DIR / "heatwave_forecast_compact.joblib"
    joblib.dump(best_model, compact_path)
    compact_meta = {
        "features": feature_columns,
        "target": target,
        "model": best_name,
        "threshold": results[best_name]["validation"]["threshold"],
        "threshold_method": "validation_max_balanced_accuracy_with_recall_constraint",
        "probability_correction": "none_full_training_prevalence",
        "climatology_fit": "train_only",
        "minimum_validation_recall": 0.75,
        "xgboost_scale_pos_weight": 7.0,
        "train_end": str(train["date"].max().date()),
        "climatology_fit": "train_only_district_month",
    }
    (MODEL_DIR / "heatwave_forecast_compact_meta.json").write_text(json.dumps(compact_meta, indent=2), encoding="utf-8")

    model_path = MODEL_DIR / f"{best_name}_{target}.joblib"
    joblib.dump(best_model, model_path)

    results_path = MODEL_DIR / f"training_results_{target}.json"
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    # Save chronological test predictions for inspection.
    predictions = test[["date", "district", target]].copy()
    predictions["predicted_probability"] = best_model.predict_proba(X_test)[:, 1]
    best_threshold = results[best_name]["validation"]["threshold"]
    predictions["predicted_class"] = (predictions["predicted_probability"] >= best_threshold).astype(int)
    predictions.to_csv(
        MODEL_DIR / f"test_predictions_{target}.csv",
        index=False,
    )
    _save_diagnostics(best_model, X_train, y_train, y_test.to_numpy(), predictions["predicted_probability"].to_numpy(), best_threshold, target)

    log("")
    log("=" * 70)
    log("BEST MODEL")
    log("=" * 70)
    log(f"Model: {best_name}")
    log(f"Saved: {model_path}")
    log(f"Results: {results_path}")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "results": results,
        "model_path": model_path,
        "results_path": results_path,
        "test_predictions_path": MODEL_DIR / f"test_predictions_{target}.csv",
    }
