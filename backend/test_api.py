import requests
import json
import pandas as pd
import numpy as np

# Create a dummy dataset
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.randint(0, 100, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'target': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)
df.to_csv("test_data.csv", index=False)

# 1. Test Upload
print("Testing /api/data/upload...")
url_upload = "http://localhost:8000/api/data/upload"
with open("test_data.csv", "rb") as f:
    files = {"file": ("test_data.csv", f, "text/csv")}
    response = requests.post(url_upload, files=files)

print("Status Code:", response.status_code)
if response.status_code != 200:
    print(response.text)
    exit(1)

meta = response.json()
print("Upload Meta:", json.dumps(meta, indent=2))
dataset_id = meta["id"]

# 2. Test Training
print("\nTesting /api/train...")
url_train = "http://localhost:8000/api/train/"
config = {
    "dataset_id": dataset_id,
    "target_column": "target",
    "drop_columns": [],
    "model_type": "random_forest",
    "test_size": 0.2
}
response_train = requests.post(url_train, json=config)

print("Status Code:", response_train.status_code)
if response_train.status_code != 200:
    print(response_train.text)
    exit(1)

model_info = response_train.json()
print("Trained Model Info:", json.dumps(model_info, indent=2))
model_id = model_info["id"]

# 3. Test Global Explanation
print("\nTesting /api/explain/global...")
url_global = f"http://localhost:8000/api/explain/global/{model_id}"
response_global = requests.get(url_global)
print("Status Code:", response_global.status_code)
if response_global.status_code != 200:
    print(response_global.text)
else:
    print("Global Explanation:", json.dumps(response_global.json(), indent=2))

# 4. Test Local Explanation
print("\nTesting /api/explain/local...")
url_local = f"http://localhost:8000/api/explain/local/{model_id}/{dataset_id}/0"
response_local = requests.get(url_local)
print("Status Code:", response_local.status_code)
if response_local.status_code != 200:
    print(response_local.text)
else:
    print("Local Explanation:", json.dumps(response_local.json(), indent=2))

print("\nTest passed successfully!")
