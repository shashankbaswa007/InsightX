"""
InsightX AI — Unit Tests for ML Service (ml_service.py)

Tests the core preprocessing, training, and metric sanitization logic
in isolation (no HTTP layer involved).
"""
import math
import numpy as np
import pandas as pd
import pytest

from app.services.ml_service import preprocess_data, _sanitize_metric, train_model
from app.models.schemas import TrainingConfig
from tests.conftest import upload_dataset


# ═══════════════════════════════════════════════════════════════════
#  preprocess_data — unit tests
# ═══════════════════════════════════════════════════════════════════

class TestPreprocessData:
    """Tests for the preprocess_data() function."""

    def test_basic_classification(self):
        """Numeric features + binary target → classification task."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        })
        X, y, is_cls = preprocess_data(df, "target", [])
        assert is_cls is True
        assert list(X.columns) == ["a", "b"]
        assert len(X) == 5
        assert len(y) == 5

    def test_regression_detected_for_many_unique(self):
        """Target with ≥20 unique float values → regression task."""
        df = pd.DataFrame({
            "a": range(30),
            "target": np.random.rand(30) * 100,
        })
        X, y, is_cls = preprocess_data(df, "target", [])
        assert is_cls is False

    def test_categorical_target_encoded(self):
        """String target column gets label-encoded for classification."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "label": ["cat", "dog", "cat", "fish", "dog", "cat"],
        })
        X, y, is_cls = preprocess_data(df, "label", [])
        assert is_cls is True
        # y should be integer-encoded
        assert y.dtype in [np.int64, np.int32, np.intp]
        assert set(y.unique()).issubset({0, 1, 2})

    def test_categorical_features_kept_raw(self):
        """Categorical features are kept raw for TargetEncoder later."""
        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["A", "B", "A", "C"],
            "target": [0, 1, 0, 1],
        })
        X, y, _ = preprocess_data(df, "target", [])
        assert "cat" in X.columns
        assert "num" in X.columns

    def test_drop_columns(self):
        """Specified columns get dropped before training."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "feat": [10, 20, 30],
            "target": [0, 1, 0],
        })
        X, y, _ = preprocess_data(df, "target", ["id", "name"])
        assert "id" not in X.columns
        assert "name" not in X.columns
        assert "feat" in X.columns

    def test_drop_columns_nonexistent_ignored(self):
        """Dropping a column that doesn't exist is silently ignored."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        X, y, _ = preprocess_data(df, "target", ["nonexistent_col"])
        assert "a" in X.columns

    def test_nan_target_rows_dropped(self):
        """Rows with NaN target values get dropped."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "target": [0, np.nan, 1, np.nan, 0],
        })
        X, y, _ = preprocess_data(df, "target", [])
        assert len(X) == 3
        assert len(y) == 3

    def test_all_nan_target_raises(self):
        """All-NaN target column raises ValueError."""
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "target": [np.nan, np.nan, np.nan],
        })
        with pytest.raises(ValueError, match="empty after dropping"):
            preprocess_data(df, "target", [])

    def test_missing_target_column_raises(self):
        """Non-existent target column raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            preprocess_data(df, "nonexistent", [])

    def test_no_features_remaining_raises(self):
        """Dropping all feature columns raises ValueError."""
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        with pytest.raises(ValueError, match="No features remaining"):
            preprocess_data(df, "target", ["a"])

    def test_numeric_nan_kept(self):
        """Missing numeric feature values are kept (handled during train_model)."""
        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0, np.nan, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        X, y, _ = preprocess_data(df, "target", [])
        assert X.isnull().any().any(), "NaN values should be kept for downstream imputer"

    def test_categorical_nan_kept(self):
        """Missing categorical feature values are kept (handled during train_model)."""
        df = pd.DataFrame({
            "cat": ["A", "B", np.nan, "A", np.nan],
            "target": [0, 1, 0, 1, 0],
        })
        X, y, _ = preprocess_data(df, "target", [])
        assert X.isnull().any().any(), "NaN values should be kept for downstream imputer"


# ═══════════════════════════════════════════════════════════════════
#  _sanitize_metric — unit tests
# ═══════════════════════════════════════════════════════════════════

class TestSanitizeMetric:
    """Tests for the _sanitize_metric() helper."""

    def test_none_passthrough(self):
        assert _sanitize_metric(None) is None

    def test_normal_float(self):
        assert _sanitize_metric(0.95) == 0.95

    def test_normal_int(self):
        assert _sanitize_metric(42) == 42

    def test_nan_becomes_none(self):
        assert _sanitize_metric(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _sanitize_metric(float("inf")) is None

    def test_neg_inf_becomes_none(self):
        assert _sanitize_metric(float("-inf")) is None

    def test_numpy_nan(self):
        assert _sanitize_metric(np.float64("nan")) is None

    def test_numpy_inf(self):
        assert _sanitize_metric(np.float64("inf")) is None

    def test_numpy_float_normal(self):
        result = _sanitize_metric(np.float64(0.5))
        assert result == 0.5
        assert isinstance(result, float)

    def test_numpy_int(self):
        result = _sanitize_metric(np.int64(7))
        assert result == 7
        assert isinstance(result, int)

    def test_list_sanitized(self):
        result = _sanitize_metric([1.0, float("nan"), 3.0])
        assert result == [1.0, None, 3.0]

    def test_dict_sanitized(self):
        result = _sanitize_metric({"a": 1.0, "b": float("inf"), "c": "hello"})
        assert result == {"a": 1.0, "b": None, "c": "hello"}

    def test_nested_dict_list(self):
        result = _sanitize_metric({
            "values": [1.0, float("nan")],
            "nested": {"x": float("-inf")},
        })
        assert result == {
            "values": [1.0, None],
            "nested": {"x": None},
        }

    def test_string_passthrough(self):
        assert _sanitize_metric("hello") == "hello"

    def test_confusion_matrix(self):
        """Confusion matrix (list of lists of ints) passes through cleanly."""
        cm = [[10, 2], [3, 15]]
        assert _sanitize_metric(cm) == [[10, 2], [3, 15]]


# ═══════════════════════════════════════════════════════════════════
#  train_model — unit tests (requires filesystem for dataset storage)
# ═══════════════════════════════════════════════════════════════════

class TestTrainModel:
    """Tests for the train_model() function using the TestClient to upload data first."""

    def test_train_classification_rf(self, client, clean_classification_csv):
        """Train a random forest classifier end-to-end at service level."""
        resp = upload_dataset(client, clean_classification_csv)
        assert resp.status_code == 200
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="target",
            drop_columns=[],
            model_type="random_forest",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.task_type == "classification"
        assert result.model_type == "random_forest"
        assert result.metrics.accuracy is not None
        assert 0.0 <= result.metrics.accuracy <= 1.0
        assert result.metrics.confusion_matrix is not None
        assert len(result.feature_names) > 0

    def test_train_classification_gb(self, client, clean_classification_csv):
        """Train a gradient boosting classifier."""
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="target",
            drop_columns=[],
            model_type="gradient_boosting",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.task_type == "classification"
        assert result.model_type == "gradient_boosting"

    def test_train_classification_logistic(self, client, clean_classification_csv):
        """Train a logistic regression classifier."""
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="target",
            drop_columns=[],
            model_type="logistic_regression",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.task_type == "classification"
        assert result.model_type == "logistic_regression"

    def test_train_regression(self, client, regression_csv):
        """Train on a regression dataset — model should switch to regression mode."""
        resp = upload_dataset(client, regression_csv, "regression.csv")
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="continuous_target",
            drop_columns=[],
            model_type="random_forest",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.task_type == "regression"
        assert result.metrics.rmse is not None
        assert result.metrics.mae is not None
        assert result.metrics.r2 is not None

    def test_train_regression_linear(self, client, regression_csv):
        """logistic_regression on regression data → linear_regression."""
        resp = upload_dataset(client, regression_csv, "regression2.csv")
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="continuous_target",
            drop_columns=[],
            model_type="logistic_regression",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.model_type == "linear_regression"
        assert result.task_type == "regression"

    def test_train_with_drop_columns(self, client, dataset_with_ids_csv):
        """Training with drop_columns excludes those features."""
        resp = upload_dataset(client, dataset_with_ids_csv, "with_ids.csv")
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="label",
            drop_columns=["id_col", "name_col"],
            model_type="random_forest",
            test_size=0.2,
        )
        result = train_model(config)
        assert "id_col" not in result.feature_names
        assert "name_col" not in result.feature_names
        assert "feature_a" in result.feature_names

    def test_train_multiclass(self, client, multiclass_csv):
        """Train on a multi-class classification dataset."""
        resp = upload_dataset(client, multiclass_csv, "multiclass.csv")
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="label",
            drop_columns=[],
            model_type="random_forest",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.task_type == "classification"
        assert result.metrics.accuracy is not None

    def test_train_nan_target_rows_handled(self, client, nan_target_csv):
        """NaN target rows are dropped; training still succeeds."""
        resp = upload_dataset(client, nan_target_csv, "nan_target.csv")
        dataset_id = resp.json()["id"]

        config = TrainingConfig(
            dataset_id=dataset_id,
            target_column="target",
            drop_columns=[],
            model_type="random_forest",
            test_size=0.2,
        )
        result = train_model(config)
        assert result.metrics.accuracy is not None

    def test_train_nonexistent_dataset_raises(self):
        """Training on a non-existent dataset ID raises FileNotFoundError."""
        config = TrainingConfig(
            dataset_id="00000000-0000-0000-0000-000000000000",
            target_column="target",
            drop_columns=[],
            model_type="random_forest",
            test_size=0.2,
        )
        with pytest.raises(FileNotFoundError):
            train_model(config)
