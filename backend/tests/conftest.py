"""
InsightX AI — Shared Test Fixtures (conftest.py)
Provides reusable pytest fixtures for the entire test suite:
  - FastAPI TestClient
  - Temp directories for uploads/models
  - Pre-built CSV datasets (clean, NaN, Inf, categorical, multiclass, regression)
  - Helper to upload a dataset and return its metadata
"""
import os
import io
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Patch config paths BEFORE importing the app — this avoids polluting real data dirs
_test_upload_dir = tempfile.mkdtemp(prefix="insightx_test_uploads_")
_test_model_dir = tempfile.mkdtemp(prefix="insightx_test_models_")

import app.config as cfg
cfg.UPLOAD_DIR = _test_upload_dir
cfg.MODEL_DIR = _test_model_dir
os.makedirs(_test_upload_dir, exist_ok=True)
os.makedirs(_test_model_dir, exist_ok=True)

from app.main import app as fastapi_app


# ─── Core Fixtures ────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Session-scoped FastAPI test client — no real server needed."""
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture(autouse=True, scope="session")
def cleanup_temp_dirs():
    """Clean up temp dirs after the full test session."""
    yield
    shutil.rmtree(_test_upload_dir, ignore_errors=True)
    shutil.rmtree(_test_model_dir, ignore_errors=True)


# ─── Dataset Builders ─────────────────────────────────────────────

def _df_to_upload_tuple(df: pd.DataFrame, filename: str = "test.csv"):
    """Convert a DataFrame to an (filename, bytes, mime) tuple for TestClient."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return (filename, buf, "text/csv")


@pytest.fixture(scope="session")
def clean_classification_csv():
    """100-row clean binary classification dataset with numeric + categorical features."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.randint(0, 100, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100),
    })
    return df


@pytest.fixture(scope="session")
def nan_target_csv():
    """Dataset with NaN values in the target column."""
    df = pd.DataFrame({
        "x1": list(range(20)),
        "x2": np.random.rand(20),
        "target": [0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0,
                   1, 0, np.nan, 1, 0, 1, 0, 1, 0, 1],
    })
    return df


@pytest.fixture(scope="session")
def inf_csv():
    """Dataset with Infinity and -Infinity values."""
    df = pd.DataFrame({
        "val": [1.0, float("inf"), -float("inf"), 4.0, np.nan],
        "target": [0, 1, 0, 1, 0],
    })
    return df


@pytest.fixture(scope="session")
def regression_csv():
    """50-row regression dataset with a continuous target (>20 unique values)."""
    np.random.seed(42)
    df = pd.DataFrame({
        "x1": np.random.rand(50),
        "x2": np.random.rand(50),
        "continuous_target": np.random.rand(50) * 100,
    })
    return df


@pytest.fixture(scope="session")
def multiclass_csv():
    """Dataset with 4-class categorical target."""
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_a": np.random.rand(60),
        "feat_b": np.random.randint(0, 50, 60),
        "label": np.random.choice(["cat", "dog", "fish", "bird"], 60),
    })
    return df


@pytest.fixture(scope="session")
def dataset_with_ids_csv():
    """Dataset with ID and name columns that should be dropped before training."""
    np.random.seed(42)
    df = pd.DataFrame({
        "id_col": range(40),
        "name_col": [f"item_{i}" for i in range(40)],
        "feature_a": np.random.rand(40),
        "feature_b": np.random.randint(0, 50, 40),
        "label": np.random.choice([0, 1], 40),
    })
    return df


# ─── Upload Helper ─────────────────────────────────────────────────

def upload_dataset(client, df: pd.DataFrame, filename: str = "test.csv"):
    """Upload a DataFrame via the API and return the response JSON."""
    file_tuple = _df_to_upload_tuple(df, filename)
    resp = client.post("/api/data/upload", files={"file": file_tuple})
    return resp
