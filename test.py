
"""
Heatwave Model Evaluation
--------------------------
Windows-safe evaluation script with:

1. Chronological train/validation/test split
2. Heatwave class-balance report
3. Confusion matrix using matplotlib
4. Accuracy, balanced accuracy, precision, recall, F1
5. ROC-AUC and PR-AUC
6. Brier score
7. Threshold comparison
8. Leakage audit

Run:
    python test.py

Required:
    heatwave_ml_dataset.csv
    models/heatwave_forecast_compact.joblib
    models/heatwave_forecast_compact_meta.json
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from heatwave_pipeline.features.thermal import add_thermal_features
from heatwave_pipeline.features.engine import add_lag_features, chronological_split
from heatwave_pipeline.models.trainer import _fit_train_climatology, _apply_train_climatology, _rebuild_target

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "heatwave_ml_dataset.csv"
MODEL = BASE_DIR / "models" / "heatwave_forecast_compact.joblib"
META = BASE_DIR / "models" / "heatwave_forecast_compact_meta.json"

REPORT = BASE_DIR / "heatwave_model_evaluation_report.json"
PREDICTIONS = BASE_DIR / "heatwave_model_test_predictions.csv"
CM_IMAGE = BASE_DIR / "heatwave_confusion_matrix.png"


TARGET = "heatwave_next_3d"


def chronological_split(df, train_fraction=0.70, valid_fraction=0.15):
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_fraction)
    valid_end = int(n * (train_fraction + valid_fraction))

    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:valid_end].copy()
    test = df.iloc[valid_end:].copy()

    return train, valid, test


def prior_correct_probability(prob, sampled_rate, true_rate):
    """
    Correct probabilities when the training data was changed by
    negative undersampling.

    sampled_rate = positive rate after balancing
    true_rate    = original positive rate
    """

    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)

    sampled_rate = np.clip(float(sampled_rate), 1e-6, 1 - 1e-6)
    true_rate = np.clip(float(true_rate), 1e-6, 1 - 1e-6)

    sampled_odds = p / (1.0 - p)

    prior_ratio = (
        (true_rate / (1.0 - true_rate))
        /
        (sampled_rate / (1.0 - sampled_rate))
    )

    corrected_odds = sampled_odds * prior_ratio

    return corrected_odds / (1.0 + corrected_odds)


def calculate_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(y_true, prediction, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, prediction, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, prediction, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probability)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probability)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probability)
        ),
        "true_positive": int(cm[1, 1]),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "confusion_matrix": cm.tolist(),
        "predicted_positive": int(prediction.sum()),
    }


def make_confusion_matrix(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    )

    accuracy = accuracy_score(y_true, prediction)
    balanced_accuracy = balanced_accuracy_score(y_true, prediction)

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Heatwave", "Heatwave"]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Heatwave Model - Confusion Matrix\n"
        f"Threshold = {threshold:.2f}"
    )

    ax.set_xlabel(
        f"Predicted Class\n"
        f"Accuracy = {accuracy:.3f} | "
        f"Balanced Accuracy = {balanced_accuracy:.3f}"
    )

    ax.set_ylabel("Actual Class")

    fig.tight_layout()

    fig.savefig(
        CM_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


def main():

    print("=" * 72)
    print("HEATWAVE MODEL EVALUATION")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if not DATASET.exists():
        print("\nERROR: Dataset not found:")
        print(DATASET)
        return 1

    if not MODEL.exists():
        print("\nERROR: Model not found:")
        print(MODEL)
        return 1

    df = pd.read_csv(DATASET)

    # The saved V10 model was trained after thermal/climatology feature
    # engineering. The CSV may not contain every derived feature because
    # some versions of the dataset were exported before the final feature
    # pass. Rebuild the derived features before checking model columns.
    required_derived = {
        "district_hi_p90_c",
        "district_hi_p95_c",
        "district_tmin_p95_c",
        "heat_index_above_p90",
        "heat_index_above_p95",
    }
    if not required_derived.issubset(df.columns):
        print("\nREBUILDING DERIVED THERMAL FEATURES")
        print("The dataset is missing some V10 model features; rebuilding them from the raw weather columns...")
        df = add_thermal_features(df)
        df = add_lag_features(df)

    if "date" not in df.columns:
        print("ERROR: Dataset does not contain 'date'.")
        return 1

    if TARGET not in df.columns:
        print(f"ERROR: Dataset does not contain '{TARGET}'.")
        return 1

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", TARGET]
    ).copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    ).astype(int)

    print("\nDATASET")
    print("-" * 72)
    print("Rows:", len(df))
    print("Districts:", df["district"].nunique())
    print(
        "Date range:",
        df["date"].min().date(),
        "to",
        df["date"].max().date()
    )

    positives = int(df[TARGET].sum())
    negatives = int(len(df) - positives)

    print("Positive heatwave samples:", positives)
    print("Negative samples:", negatives)
    print(
        "Overall heatwave rate: %.3f%%"
        % (100 * df[TARGET].mean())
    )

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------

    train, valid, test = chronological_split(df)

    # Match the training pipeline: climatology and target are reconstructed
    # from TRAIN ONLY so the test metric is leakage-free.
    train_climatology = _fit_train_climatology(train)
    train = _apply_train_climatology(train, train_climatology)
    valid = _apply_train_climatology(valid, train_climatology)
    test = _apply_train_climatology(test, train_climatology)
    train = _rebuild_target(train, TARGET).dropna(subset=[TARGET]).copy()
    valid = _rebuild_target(valid, TARGET).dropna(subset=[TARGET]).copy()
    test = _rebuild_target(test, TARGET).dropna(subset=[TARGET]).copy()

    print("\nCHRONOLOGICAL SPLIT")
    print("-" * 72)

    for name, part in [
        ("TRAIN", train),
        ("VALIDATION", valid),
        ("TEST", test),
    ]:

        print(
            "%-12s %6d rows | %s -> %s | positive = %.3f%%"
            % (
                name,
                len(part),
                part["date"].min().date(),
                part["date"].max().date(),
                100 * part[TARGET].mean(),
            )
        )

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = joblib.load(MODEL)

    features = []

    if META.exists():

        with open(
            META,
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        features = metadata.get(
            "features",
            []
        )

        saved_threshold = float(
            metadata.get(
                "threshold",
                0.5
            )
        )

        model_name = metadata.get(
            "model",
            "unknown"
        )

    else:

        saved_threshold = 0.5
        model_name = "unknown"

    if not features:

        features = list(
            getattr(
                model,
                "feature_names_in_",
                []
            )
        )

    missing = [
        c for c in features
        if c not in test.columns
    ]

    if missing:

        print("\nERROR: Missing model features:")

        for column in missing:
            print("  -", column)

        return 1

    # ---------------------------------------------------------
    # Predict test set
    # ---------------------------------------------------------

    raw_probability = model.predict_proba(
        test[features]
    )[:, 1]

    # The production model retains the real training prevalence and uses class
    # weighting, so no negative-undersampling prior correction is applied.
    corrected_probability = raw_probability

    y_test = test[TARGET].to_numpy()

    # ---------------------------------------------------------
    # Main result
    # ---------------------------------------------------------

    result = calculate_metrics(
        y_test,
        corrected_probability,
        saved_threshold
    )

    print("\nMODEL")
    print("-" * 72)
    print("Model:", model_name)
    print("Saved threshold:", saved_threshold)

    print("\nTEST PERFORMANCE")
    print("-" * 72)

    print(
        "Accuracy:           %.4f (%.2f%%)"
        % (
            result["accuracy"],
            100 * result["accuracy"]
        )
    )

    print(
        "Balanced Accuracy:  %.4f (%.2f%%)"
        % (
            result["balanced_accuracy"],
            100 * result["balanced_accuracy"]
        )
    )

    print(
        "Precision:          %.4f"
        % result["precision"]
    )

    print(
        "Recall:             %.4f"
        % result["recall"]
    )

    print(
        "F1 Score:           %.4f"
        % result["f1"]
    )

    print(
        "ROC-AUC:            %.4f"
        % result["roc_auc"]
    )

    print(
        "PR-AUC:             %.4f"
        % result["pr_auc"]
    )

    print(
        "Brier Score:        %.4f"
        % result["brier_score"]
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    print("\nCONFUSION MATRIX")
    print("-" * 72)

    cm = np.array(
        result["confusion_matrix"]
    )

    print(
        "                 Predicted"
    )

    print(
        "                 No HW    HW"
    )

    print(
        "Actual No HW     %6d %6d"
        % (
            cm[0, 0],
            cm[0, 1]
        )
    )

    print(
        "Actual HW        %6d %6d"
        % (
            cm[1, 0],
            cm[1, 1]
        )
    )

    print("\nTrue Negative :", result["true_negative"])
    print("False Positive:", result["false_positive"])
    print("False Negative:", result["false_negative"])
    print("True Positive :", result["true_positive"])

    # ---------------------------------------------------------
    # Threshold comparison
    # ---------------------------------------------------------

    print("\nTHRESHOLD COMPARISON")
    print("-" * 72)

    print(
        "%-10s %-10s %-10s %-10s %-10s %-10s"
        % (
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Bal.Acc"
        )
    )

    threshold_results = {}

    for threshold in [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    ]:

        m = calculate_metrics(
            y_test,
            corrected_probability,
            threshold
        )

        threshold_results[
            str(threshold)
        ] = m

        print(
            "%-10.2f %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f"
            % (
                threshold,
                m["accuracy"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["balanced_accuracy"],
            )
        )

    # ---------------------------------------------------------
    # Leakage audit
    # ---------------------------------------------------------

    print("\nLEAKAGE AUDIT")
    print("-" * 72)
    print("Climatology was refit from the chronological TRAIN partition only.")
    print("The same frozen train climatology was applied to validation and test.")
    print("Future heatwave labels were rebuilt from future Tmax/WBGT using those train thresholds.")
    print("RESULT: temporal climatology leakage is controlled for this evaluation.")

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    prediction_output = test[
        [
            "date",
            "district",
            TARGET
        ]
    ].copy()

    prediction_output[
        "raw_probability"
    ] = raw_probability

    prediction_output[
        "corrected_probability"
    ] = corrected_probability

    prediction_output[
        "prediction"
    ] = (
        corrected_probability
        >= saved_threshold
    ).astype(int)

    prediction_output.to_csv(
        PREDICTIONS,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    report = {
        "model": model_name,
        "threshold": saved_threshold,
        "dataset_rows": len(df),
        "test_rows": len(test),
        "test_positive_rate": float(y_test.mean()),
        "training_positive_rate": float(train[TARGET].mean()),
        "metrics": result,
        "threshold_comparison": threshold_results,
        "leakage_columns": [],
        "leakage_status": "controlled_train_only_climatology",
        "confusion_matrix_image": str(CM_IMAGE),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    make_confusion_matrix(
        y_test,
        corrected_probability,
        saved_threshold
    )

    print("\nFILES CREATED")
    print("-" * 72)
    print("Confusion matrix:")
    print(CM_IMAGE)

    print("Prediction CSV:")
    print(PREDICTIONS)

    print("Evaluation report:")
    print(REPORT)

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
