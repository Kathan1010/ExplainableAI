import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'student_data.csv')

_df_cache = None


def _load_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DATA_PATH)
    return _df_cache


def get_summary():
    df = _load_df()
    return {
        "total_records": len(df),
        "feature_count": len(df.columns) - 1,
        "class_count": int(df["target"].nunique()),
        "class_distribution": df["target"].value_counts().to_dict(),
    }


def get_preview(n=10):
    df = _load_df()
    return df.head(n).to_dict(orient="records")


def get_target_distribution():
    df = _load_df()
    counts = df["target"].value_counts()
    return {"labels": counts.index.tolist(), "values": counts.values.tolist()}


def get_gpa_by_target():
    df = _load_df()
    result = []
    for target in df["target"].unique():
        subset = df[df["target"] == target]["prev_gpa"]
        result.append({
            "target": target,
            "values": subset.tolist(),
        })
    return result


def get_correlation_matrix():
    df = _load_df()
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()
    return {
        "labels": corr.columns.tolist(),
        "matrix": np.round(corr.values, 3).tolist(),
    }
