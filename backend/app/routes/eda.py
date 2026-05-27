"""
EDA (Exploratory Data Analysis) endpoint.
Returns per-column statistics, missing value summary, and histograms.
"""

import os
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import UPLOAD_DIR

router = APIRouter(prefix="/api/data", tags=["EDA"])


def _safe_float(val):
    """Convert numpy floats to Python floats, handling NaN/Inf."""
    if val is None:
        return None
    if isinstance(val, (np.floating, float)):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    if isinstance(val, (np.integer,)):
        return int(val)
    return val


@router.get("/{dataset_id}/eda")
async def get_eda(dataset_id: str):
    """
    Comprehensive EDA for a dataset.
    Returns column stats, missing values, histograms, and categorical counts.
    """
    # Find file
    file_path = None
    for ext in [".csv", ".json"]:
        p = os.path.join(UPLOAD_DIR, f"{dataset_id}{ext}")
        if os.path.exists(p):
            file_path = p
            break

    if not file_path:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_json(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {e}")

    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing = int(df.isnull().sum().sum())
    total_cells = total_rows * total_cols
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0
    duplicate_rows = int(df.duplicated().sum())

    # ── Per-column stats ──────────────────────────────────────
    column_stats = []
    histograms = []
    categorical_counts = []

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0
        unique_count = int(df[col].nunique())

        entry = {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            entry["mean"] = _safe_float(series.mean())
            entry["median"] = _safe_float(series.median())
            entry["std"] = _safe_float(series.std())
            entry["min"] = _safe_float(series.min())
            entry["max"] = _safe_float(series.max())
            entry["skewness"] = _safe_float(series.skew())
            entry["type"] = "numeric"

            # Histogram (20 bins)
            if len(series) > 0:
                counts, bin_edges = np.histogram(series.values, bins=20)
                histograms.append({
                    "column": str(col),
                    "bins": [round(float(b), 4) for b in bin_edges],
                    "counts": [int(c) for c in counts],
                })
        else:
            entry["type"] = "categorical"
            # Top 10 value counts
            vc = df[col].value_counts().head(10)
            top_values = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
            entry["top_values"] = top_values

            categorical_counts.append({
                "column": str(col),
                "values": top_values,
            })

        column_stats.append(entry)

    # ── Missing value summary (sorted by % descending) ────────
    missing_summary = [
        {"column": s["name"], "count": s["null_count"], "pct": s["null_pct"]}
        for s in sorted(column_stats, key=lambda x: x["null_pct"], reverse=True)
        if s["null_count"] > 0
    ]

    return {
        "summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_missing_cells": total_missing,
            "missing_pct": missing_pct,
            "duplicate_rows": duplicate_rows,
        },
        "column_stats": column_stats,
        "missing_summary": missing_summary,
        "histograms": histograms,
        "categorical_counts": categorical_counts,
    }
