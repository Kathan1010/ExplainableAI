import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

_models = None
_preprocessor = None
_feature_names = None


def _load_models():
    global _models
    if _models is None:
        _models = joblib.load(os.path.join(MODEL_DIR, 'models.joblib'))
    return _models


def _load_preprocessor():
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
    return _preprocessor


def _load_feature_names():
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    return _feature_names


def get_model(name="XGBoost"):
    return _load_models()[name]


def get_preprocessor():
    return _load_preprocessor()


def get_feature_names():
    return _load_feature_names()


def load_test_data():
    X_test = pd.read_csv(os.path.join(MODEL_DIR, 'X_test_processed.csv'))
    y_test = pd.read_csv(os.path.join(MODEL_DIR, 'y_test.csv')).squeeze('columns')
    return X_test, y_test


def evaluate_all_models():
    X_test, y_test = load_test_data()
    models = _load_models()

    results = {}
    confusion_matrices = {}
    roc_data = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision (Weighted)": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "Recall (Weighted)": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "F1 Score (Weighted)": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        }

        if y_prob is not None:
            metrics["AUC-ROC (OVR)"] = round(
                roc_auc_score(y_test, y_prob, multi_class='ovr'), 4
            )
            roc_data[name] = {}
            for i in range(3):
                y_test_bin = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(y_test_bin, y_prob[:, i])
                roc_data[name][str(i)] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }

        results[name] = metrics
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        confusion_matrices[name] = cm.tolist()

    return {
        "metrics": results,
        "confusion_matrices": confusion_matrices,
        "roc_curves": roc_data,
    }


def predict_student(input_data: dict):
    preprocessor = _load_preprocessor()
    feature_names = _load_feature_names()
    model = get_model("XGBoost")

    df = pd.DataFrame([input_data])
    processed = preprocessor.transform(df)
    processed_df = pd.DataFrame(processed, columns=feature_names)

    pred_idx = int(model.predict(processed_df)[0])
    pred_probs = model.predict_proba(processed_df)[0].tolist()
    class_names = ["Fail", "At-Risk", "Pass"]

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "probabilities": {cn: round(p, 4) for cn, p in zip(class_names, pred_probs)},
        "confidence": round(max(pred_probs) * 100, 1),
        "processed_features": processed_df.iloc[0].to_dict(),
    }
