import shap
import numpy as np
import pandas as pd
from .model_service import get_model, load_test_data


def get_global_shap(model_name="XGBoost"):
    """Return mean absolute SHAP values per feature for global interpretation."""
    model = get_model(model_name)
    X_test, _ = load_test_data()

    if model_name in ["XGBoost", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        background = shap.sample(X_test, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test.iloc[:100])

    # shap_values is list of arrays (one per class) for multiclass
    if isinstance(shap_values, list):
        combined = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
    else:
        combined = np.abs(shap_values).mean(axis=0)
        if combined.ndim > 1:
            combined = combined.mean(axis=1)

    feature_names = X_test.columns.tolist()
    sorted_indices = np.argsort(combined)[::-1]

    return {
        "features": [feature_names[i] for i in sorted_indices],
        "importance": [round(float(combined[i]), 4) for i in sorted_indices],
    }


def get_local_shap(student_index: int, model_name="XGBoost"):
    """Return per-feature SHAP contributions for a single student."""
    model = get_model(model_name)
    X_test, _ = load_test_data()

    student_data = X_test.iloc[[student_index]]

    if model_name in ["XGBoost", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_obj = explainer(student_data)
    else:
        background = shap.sample(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_obj = explainer(student_data)

    pred_idx = int(model.predict(student_data)[0])
    class_names = ["Fail", "At-Risk", "Pass"]

    # Extract values for the predicted class
    if shap_obj.values.ndim == 3:
        values = shap_obj.values[0, :, pred_idx].tolist()
        base_value = float(shap_obj.base_values[0, pred_idx])
    else:
        values = shap_obj.values[0].tolist()
        base_value = float(shap_obj.base_values[0])

    feature_names = X_test.columns.tolist()
    feature_values = student_data.iloc[0].tolist()

    return {
        "predicted_class": class_names[pred_idx],
        "predicted_index": pred_idx,
        "base_value": round(base_value, 4),
        "features": feature_names,
        "shap_values": [round(v, 4) for v in values],
        "feature_values": [round(v, 4) for v in feature_values],
    }


def get_shap_for_input(processed_features: dict, model_name="XGBoost"):
    """SHAP explanation for a live prediction (already-processed features)."""
    model = get_model(model_name)
    X_test, _ = load_test_data()

    student_df = pd.DataFrame([processed_features])

    if model_name in ["XGBoost", "Decision Tree"]:
        explainer = shap.TreeExplainer(model)
        shap_obj = explainer(student_df)
    else:
        background = shap.sample(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_obj = explainer(student_df)

    pred_idx = int(model.predict(student_df)[0])

    if shap_obj.values.ndim == 3:
        values = shap_obj.values[0, :, pred_idx].tolist()
        base_value = float(shap_obj.base_values[0, pred_idx])
    else:
        values = shap_obj.values[0].tolist()
        base_value = float(shap_obj.base_values[0])

    feature_names = list(processed_features.keys())

    return {
        "base_value": round(base_value, 4),
        "features": feature_names,
        "shap_values": [round(v, 4) for v in values],
        "feature_values": [round(float(v), 4) for v in student_df.iloc[0].tolist()],
    }
