"""
InsightX - Edge Case Tests
Tests for robustness issues found during the backend audit.
"""
import requests
import json
import pandas as pd
import numpy as np
import io
import os

BASE_URL = "http://localhost:8000"
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1


# ─── Test 1: Upload with NaN values in target ───
print("\n═══ Test 1: Dataset with NaN in target column ═══")
data = {
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
    'target': [0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0,
               1, 0, np.nan, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_bytes = csv_buffer.getvalue().encode()

files = {"file": ("nan_target.csv", csv_bytes, "text/csv")}
r = requests.post(f"{BASE_URL}/api/data/upload", files=files)
test("Upload with NaN target succeeds", r.status_code == 200)

if r.status_code == 200:
    meta = r.json()
    dataset_id = meta["id"]
    
    # Now train — NaN rows should be dropped silently
    config = {
        "dataset_id": dataset_id,
        "target_column": "target",
        "drop_columns": [],
        "model_type": "random_forest",
        "test_size": 0.3
    }
    r2 = requests.post(f"{BASE_URL}/api/train/", json=config)
    test("Training with NaN target rows succeeds (rows dropped)", r2.status_code == 200)
    
    if r2.status_code == 200:
        model_info = r2.json()
        test("Model has valid metrics", model_info["metrics"]["accuracy"] is not None)
        
        # Test local explanation
        r3 = requests.get(f"{BASE_URL}/api/explain/local/{model_info['id']}/{dataset_id}/0")
        test("Local explanation works after NaN-target training", r3.status_code == 200)


# ─── Test 2: Training with drop_columns + explanation ───
print("\n═══ Test 2: drop_columns preserved through explanation ═══")
data2 = {
    'id_col': range(30),
    'name_col': [f"item_{i}" for i in range(30)],
    'feature_a': np.random.rand(30),
    'feature_b': np.random.randint(0, 50, 30),
    'label': np.random.choice([0, 1], 30)
}
df2 = pd.DataFrame(data2)
csv_buf2 = io.StringIO()
df2.to_csv(csv_buf2, index=False)

files2 = {"file": ("with_ids.csv", csv_buf2.getvalue().encode(), "text/csv")}
r = requests.post(f"{BASE_URL}/api/data/upload", files=files2)
test("Upload with ID columns succeeds", r.status_code == 200)

if r.status_code == 200:
    meta2 = r.json()
    dataset_id2 = meta2["id"]
    
    config2 = {
        "dataset_id": dataset_id2,
        "target_column": "label",
        "drop_columns": ["id_col", "name_col"],  # Drop non-feature columns
        "model_type": "gradient_boosting",
        "test_size": 0.3
    }
    r2 = requests.post(f"{BASE_URL}/api/train/", json=config2)
    test("Training with dropped columns succeeds", r2.status_code == 200)
    
    if r2.status_code == 200:
        model_info2 = r2.json()
        test("Feature names exclude dropped columns", 
             "id_col" not in model_info2["feature_names"] and "name_col" not in model_info2["feature_names"],
             f"Got features: {model_info2['feature_names']}")
        
        # Critical: Local explanation must use the same drop_columns
        r3 = requests.get(f"{BASE_URL}/api/explain/local/{model_info2['id']}/{dataset_id2}/0")
        test("Local explanation works with drop_columns (no feature mismatch)", r3.status_code == 200)
        
        if r3.status_code == 200:
            local_exp = r3.json()
            test("SHAP values returned", len(local_exp["shap_values"]) > 0, f"Got {len(local_exp['shap_values'])} SHAP values")
            test("LIME explanations returned", len(local_exp["lime_explanation"]) > 0)


# ─── Test 3: Invalid model_type rejected ───
print("\n═══ Test 3: Invalid model_type validation ═══")
config3 = {
    "dataset_id": dataset_id if 'dataset_id' in dir() else "fake",
    "target_column": "target",
    "drop_columns": [],
    "model_type": "randm_forest_typo",
    "test_size": 0.2
}
r = requests.post(f"{BASE_URL}/api/train/", json=config3)
test("Invalid model_type returns 422", r.status_code == 422, f"Got {r.status_code}: {r.text[:200]}")


# ─── Test 4: Empty features after dropping all columns ───
print("\n═══ Test 4: All features dropped ═══")
if 'dataset_id' in dir():
    config4 = {
        "dataset_id": dataset_id,
        "target_column": "target",
        "drop_columns": ["feature1", "feature2"],  # Drop all features
        "model_type": "random_forest",
        "test_size": 0.2
    }
    r = requests.post(f"{BASE_URL}/api/train/", json=config4)
    test("All features dropped returns 400", r.status_code == 400, f"Got {r.status_code}")


# ─── Test 5: Infinity values in preview ───
print("\n═══ Test 5: Infinity values in data ═══")
data5 = {
    'val': [1.0, float('inf'), -float('inf'), 4.0, np.nan],
    'target': [0, 1, 0, 1, 0]
}
df5 = pd.DataFrame(data5)
csv_buf5 = io.StringIO()
df5.to_csv(csv_buf5, index=False)

files5 = {"file": ("inf_data.csv", csv_buf5.getvalue().encode(), "text/csv")}
r = requests.post(f"{BASE_URL}/api/data/upload", files=files5)
test("Upload with Inf values succeeds", r.status_code == 200)

if r.status_code == 200:
    inf_id = r.json()["id"]
    r2 = requests.get(f"{BASE_URL}/api/data/{inf_id}/preview?rows=5")
    test("Preview with Inf values returns valid JSON", r2.status_code == 200)
    
    if r2.status_code == 200:
        preview = r2.json()
        # Check that Inf values were replaced with None
        inf_vals = [row.get("val") for row in preview]
        test("Inf values replaced with None in preview", 
             float('inf') not in inf_vals and -float('inf') not in inf_vals,
             f"Got values: {inf_vals}")


# ─── Test 6: Preview row cap ───
print("\n═══ Test 6: Preview row cap ═══")
if 'dataset_id' in dir():
    r = requests.get(f"{BASE_URL}/api/data/{dataset_id}/preview?rows=99999")
    test("Huge row count doesn't crash", r.status_code == 200)


# ─── Test 7: Regression task with logistic_regression model type ───
print("\n═══ Test 7: Regression with logistic_regression (→ linear_regression) ═══")
data7 = {
    'x1': np.random.rand(50),
    'x2': np.random.rand(50),
    'continuous_target': np.random.rand(50) * 100,
}
df7 = pd.DataFrame(data7)
csv_buf7 = io.StringIO()
df7.to_csv(csv_buf7, index=False)

files7 = {"file": ("regression_data.csv", csv_buf7.getvalue().encode(), "text/csv")}
r = requests.post(f"{BASE_URL}/api/data/upload", files=files7)
if r.status_code == 200:
    reg_id = r.json()["id"]
    config7 = {
        "dataset_id": reg_id,
        "target_column": "continuous_target",
        "drop_columns": [],
        "model_type": "logistic_regression",
        "test_size": 0.2
    }
    r2 = requests.post(f"{BASE_URL}/api/train/", json=config7)
    test("Regression with logistic_regression succeeds", r2.status_code == 200)
    if r2.status_code == 200:
        test("Model type renamed to linear_regression", r2.json()["model_type"] == "linear_regression", f"Got: {r2.json()['model_type']}")
        # Verify explanation works for linear regression model
        model_id7 = r2.json()["id"]
        r3 = requests.get(f"{BASE_URL}/api/explain/global/{model_id7}")
        test("Global explanation for linear regression works", r3.status_code == 200)


# ─── Summary ───
print(f"\n{'═' * 50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
if failed == 0:
    print("🎉 All tests passed!")
else:
    print(f"⚠️  {failed} test(s) failed")
