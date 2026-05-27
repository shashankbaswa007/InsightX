"""
InsightX AI — Integration Tests for API Endpoints

Tests the complete HTTP API workflow: upload → train → explain.
Uses FastAPI's TestClient (no real server needed).
"""
import io
import json
import pytest
import pandas as pd
import numpy as np

from tests.conftest import upload_dataset


# ═══════════════════════════════════════════════════════════════════
#  /health endpoint
# ═══════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "InsightX" in data["service"]


# ═══════════════════════════════════════════════════════════════════
#  /api/data — Upload & Preview endpoints
# ═══════════════════════════════════════════════════════════════════

class TestUploadEndpoint:

    def test_upload_csv_success(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv, "clean.csv")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "clean.csv"
        assert data["rows"] == 100
        assert len(data["columns"]) == 4
        assert data["id"]  # non-empty UUID
        assert data["uploaded_at"]

    def test_upload_returns_column_metadata(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        data = resp.json()
        col_names = [c["name"] for c in data["columns"]]
        assert "feature1" in col_names
        assert "target" in col_names
        for col in data["columns"]:
            assert "dtype" in col
            assert "unique_count" in col
            assert "null_count" in col
            assert "sample_values" in col
            assert len(col["sample_values"]) <= 5

    def test_upload_json_file(self, client):
        """Upload a .json file instead of .csv."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        buf = io.BytesIO()
        df.to_json(buf, orient="records")
        buf.seek(0)
        resp = client.post("/api/data/upload", files={"file": ("data.json", buf, "application/json")})
        assert resp.status_code == 200
        assert resp.json()["rows"] == 3

    def test_upload_rejects_unsupported_extension(self, client):
        """Uploading a .txt file should return 400."""
        buf = io.BytesIO(b"some text content")
        resp = client.post("/api/data/upload", files={"file": ("data.txt", buf, "text/plain")})
        assert resp.status_code == 400
        assert "supported" in resp.json()["detail"].lower()

    def test_upload_rejects_xlsx(self, client):
        buf = io.BytesIO(b"fake excel content")
        resp = client.post("/api/data/upload", files={"file": ("data.xlsx", buf, "application/xlsx")})
        assert resp.status_code == 400

    def test_upload_with_nan_values(self, client, nan_target_csv):
        resp = upload_dataset(client, nan_target_csv, "nans.csv")
        assert resp.status_code == 200
        # Should have null_count > 0 for target column
        target_col = next(c for c in resp.json()["columns"] if c["name"] == "target")
        assert target_col["null_count"] == 2

    def test_upload_with_inf_values(self, client, inf_csv):
        """Infinity values in upload shouldn't crash JSON serialization."""
        resp = upload_dataset(client, inf_csv, "inf.csv")
        assert resp.status_code == 200
        # sample_values should not contain inf — they should be None
        val_col = next(c for c in resp.json()["columns"] if c["name"] == "val")
        for sample in val_col["sample_values"]:
            if sample is not None:
                assert sample != float("inf")


class TestPreviewEndpoint:

    def test_preview_default_rows(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]
        preview = client.get(f"/api/data/{dataset_id}/preview")
        assert preview.status_code == 200
        assert len(preview.json()) == 10  # default

    def test_preview_custom_rows(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]
        preview = client.get(f"/api/data/{dataset_id}/preview?rows=5")
        assert preview.status_code == 200
        assert len(preview.json()) == 5

    def test_preview_capped_at_500(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]
        preview = client.get(f"/api/data/{dataset_id}/preview?rows=99999")
        assert preview.status_code == 200
        # Should return at most 100 rows (the dataset has 100 rows, cap is 500)
        assert len(preview.json()) <= 100

    def test_preview_nonexistent_dataset(self, client):
        resp = client.get("/api/data/00000000-0000-0000-0000-000000000000/preview")
        assert resp.status_code == 404

    def test_preview_inf_values_sanitized(self, client, inf_csv):
        resp = upload_dataset(client, inf_csv, "inf_preview.csv")
        dataset_id = resp.json()["id"]
        preview = client.get(f"/api/data/{dataset_id}/preview?rows=5")
        assert preview.status_code == 200
        data = preview.json()
        # Check no Inf values — they should be None
        for row in data:
            val = row.get("val")
            if val is not None:
                assert val != float("inf")
                assert val != float("-inf")


# ═══════════════════════════════════════════════════════════════════
#  /api/train — Training endpoint
# ═══════════════════════════════════════════════════════════════════

class TestTrainEndpoint:

    def test_train_random_forest(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        data = train_resp.json()
        assert data["model_type"] == "random_forest"
        assert data["task_type"] == "classification"
        assert data["metrics"]["accuracy"] is not None
        assert data["metrics"]["f1_score"] is not None
        assert data["metrics"]["confusion_matrix"] is not None
        assert data["metrics"]["classification_report"] is not None

    def test_train_gradient_boosting(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "gradient_boosting",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        assert train_resp.json()["model_type"] == "gradient_boosting"

    def test_train_logistic_regression(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "logistic_regression",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200

    def test_train_regression_task(self, client, regression_csv):
        resp = upload_dataset(client, regression_csv, "reg.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "continuous_target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        data = train_resp.json()
        assert data["task_type"] == "regression"
        assert data["metrics"]["rmse"] is not None
        assert data["metrics"]["mae"] is not None
        assert data["metrics"]["r2"] is not None
        # Classification metrics should be absent
        assert data["metrics"]["accuracy"] is None
        assert data["metrics"]["confusion_matrix"] is None

    def test_train_logistic_on_regression_becomes_linear(self, client, regression_csv):
        resp = upload_dataset(client, regression_csv, "reg_lr.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "continuous_target",
            "drop_columns": [],
            "model_type": "logistic_regression",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        assert train_resp.json()["model_type"] == "linear_regression"

    def test_train_with_drop_columns(self, client, dataset_with_ids_csv):
        resp = upload_dataset(client, dataset_with_ids_csv, "ids.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "label",
            "drop_columns": ["id_col", "name_col"],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        features = train_resp.json()["feature_names"]
        assert "id_col" not in features
        assert "name_col" not in features

    def test_train_invalid_model_type_422(self, client, clean_classification_csv):
        """Invalid model_type should be rejected by Pydantic validation."""
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "invalid_model",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 422

    def test_train_nonexistent_dataset_404(self, client):
        train_resp = client.post("/api/train/", json={
            "dataset_id": "00000000-0000-0000-0000-000000000000",
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 404

    def test_train_invalid_target_column_400(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "nonexistent_column",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 400
        assert "not found" in train_resp.json()["detail"].lower()

    def test_train_all_features_dropped_400(self, client, nan_target_csv):
        resp = upload_dataset(client, nan_target_csv, "all_drop.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": ["x1", "x2"],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 400
        assert "no features" in train_resp.json()["detail"].lower()

    def test_train_invalid_test_size_422(self, client, clean_classification_csv):
        """test_size outside [0.05, 0.5] should be rejected."""
        resp = upload_dataset(client, clean_classification_csv)
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.95,
        })
        assert train_resp.status_code == 422

    def test_train_nan_target_handled(self, client, nan_target_csv):
        resp = upload_dataset(client, nan_target_csv, "nan_train.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200

    def test_train_multiclass(self, client, multiclass_csv):
        resp = upload_dataset(client, multiclass_csv, "multi.csv")
        dataset_id = resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "label",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        assert train_resp.json()["task_type"] == "classification"


# ═══════════════════════════════════════════════════════════════════
#  /api/explain — Global & Local Explanation endpoints
# ═══════════════════════════════════════════════════════════════════

class TestExplainEndpoints:

    @pytest.fixture(scope="class")
    def trained_model_ids(self, client, clean_classification_csv):
        """Upload, train, and return (dataset_id, model_id) for reuse."""
        resp = upload_dataset(client, clean_classification_csv, "explain_test.csv")
        dataset_id = resp.json()["id"]
        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        model_id = train_resp.json()["id"]
        return dataset_id, model_id

    def test_global_explanation(self, client, trained_model_ids):
        dataset_id, model_id = trained_model_ids
        resp = client.get(f"/api/explain/global/{model_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        # Each item has feature + importance
        for item in data:
            assert "feature" in item
            assert "importance" in item
            assert isinstance(item["importance"], float)
            assert item["importance"] >= 0.0
        # Should be sorted by importance descending
        importances = [item["importance"] for item in data]
        assert importances == sorted(importances, reverse=True)

    def test_global_explanation_nonexistent_model_404(self, client):
        resp = client.get("/api/explain/global/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_local_explanation(self, client, trained_model_ids):
        dataset_id, model_id = trained_model_ids
        resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["row_index"] == 0
        assert "prediction" in data
        assert "shap_values" in data
        assert "lime_explanation" in data
        # SHAP values should have correct structure
        if len(data["shap_values"]) > 0:
            sv = data["shap_values"][0]
            assert "feature" in sv
            assert "shap_value" in sv
            assert "feature_value" in sv
        # LIME explanations should have correct structure
        if len(data["lime_explanation"]) > 0:
            le = data["lime_explanation"][0]
            assert "feature" in le
            assert "weight" in le
            assert "condition" in le

    def test_local_explanation_has_probability(self, client, trained_model_ids):
        dataset_id, model_id = trained_model_ids
        resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        data = resp.json()
        # Classification models should return probability
        assert data["probability"] is not None
        assert 0.0 <= data["probability"] <= 1.0

    def test_local_explanation_multiple_rows(self, client, trained_model_ids):
        """Request explanations for different row indices."""
        dataset_id, model_id = trained_model_ids
        for row_idx in [0, 1, 5]:
            resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/{row_idx}")
            assert resp.status_code == 200
            assert resp.json()["row_index"] == row_idx

    def test_local_explanation_invalid_row_400(self, client, trained_model_ids):
        dataset_id, model_id = trained_model_ids
        resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/99999")
        assert resp.status_code == 400

    def test_local_explanation_nonexistent_model_404(self, client, clean_classification_csv):
        resp = upload_dataset(client, clean_classification_csv, "local_404.csv")
        dataset_id = resp.json()["id"]
        resp = client.get(f"/api/explain/local/00000000-0000-0000-0000-000000000000/{dataset_id}/0")
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════════
#  Full Pipeline Integration (Upload → Train → Global → Local)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end tests that exercise the complete workflow."""

    def test_classification_pipeline(self, client, clean_classification_csv):
        """Full pipeline: upload → train (RF) → global → local."""
        # 1. Upload
        upload_resp = upload_dataset(client, clean_classification_csv, "pipeline_cls.csv")
        assert upload_resp.status_code == 200
        dataset_id = upload_resp.json()["id"]

        # 2. Train
        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["id"]

        # 3. Global explanation
        global_resp = client.get(f"/api/explain/global/{model_id}")
        assert global_resp.status_code == 200
        assert len(global_resp.json()) > 0

        # 4. Local explanation
        local_resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert local_resp.status_code == 200
        local_data = local_resp.json()
        assert len(local_data["shap_values"]) > 0
        assert len(local_data["lime_explanation"]) > 0

    def test_regression_pipeline(self, client, regression_csv):
        """Full pipeline for a regression task."""
        upload_resp = upload_dataset(client, regression_csv, "pipeline_reg.csv")
        assert upload_resp.status_code == 200
        dataset_id = upload_resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "continuous_target",
            "drop_columns": [],
            "model_type": "gradient_boosting",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["id"]
        assert train_resp.json()["task_type"] == "regression"

        global_resp = client.get(f"/api/explain/global/{model_id}")
        assert global_resp.status_code == 200

        local_resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert local_resp.status_code == 200
        assert local_resp.json()["probability"] is None  # regression = no probability

    def test_drop_columns_pipeline(self, client, dataset_with_ids_csv):
        """Pipeline with drop_columns — ensure explanations use same preprocessing."""
        upload_resp = upload_dataset(client, dataset_with_ids_csv, "pipeline_drop.csv")
        dataset_id = upload_resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "label",
            "drop_columns": ["id_col", "name_col"],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["id"]
        trained_features = train_resp.json()["feature_names"]

        # Local explanation should succeed (same preprocessing as training)
        local_resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert local_resp.status_code == 200

        # SHAP features should match training features
        shap_features = [sv["feature"] for sv in local_resp.json()["shap_values"]]
        for feat in shap_features:
            assert feat in trained_features

    def test_multiclass_pipeline(self, client, multiclass_csv):
        """Full pipeline for multi-class classification."""
        upload_resp = upload_dataset(client, multiclass_csv, "pipeline_multi.csv")
        dataset_id = upload_resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "label",
            "drop_columns": [],
            "model_type": "random_forest",
            "test_size": 0.2,
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["id"]

        global_resp = client.get(f"/api/explain/global/{model_id}")
        assert global_resp.status_code == 200

        local_resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert local_resp.status_code == 200

    def test_nan_data_pipeline(self, client, nan_target_csv):
        """Full pipeline with NaN target values — should survive the whole flow."""
        upload_resp = upload_dataset(client, nan_target_csv, "pipeline_nan.csv")
        dataset_id = upload_resp.json()["id"]

        train_resp = client.post("/api/train/", json={
            "dataset_id": dataset_id,
            "target_column": "target",
            "drop_columns": [],
            "model_type": "gradient_boosting",
            "test_size": 0.3,
        })
        assert train_resp.status_code == 200
        model_id = train_resp.json()["id"]

        global_resp = client.get(f"/api/explain/global/{model_id}")
        assert global_resp.status_code == 200

        local_resp = client.get(f"/api/explain/local/{model_id}/{dataset_id}/0")
        assert local_resp.status_code == 200
