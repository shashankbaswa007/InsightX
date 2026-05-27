import os
from app.services.ml_service import train_model, preprocess_data, load_dataset
from app.models.schemas import TrainingConfig
from app.config import UPLOAD_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def verify_accuracy():
    # Find a dataset
    datasets = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.csv')]
    if not datasets:
        print("No datasets found.")
        return
        
    dataset_id = datasets[0].split('.')[0]
    df = load_dataset(dataset_id)
    target = df.columns[-1] # pick last column as target for test
    
    config = TrainingConfig(
        dataset_id=dataset_id,
        target_column=target,
        drop_columns=[],
        model_type='random_forest',
        test_size=0.2
    )
    
    # Train via service
    result = train_model(config)
    print(f"Service accuracy: {result.metrics.accuracy}")
    
    # Train manually to verify
    X, y, is_classification = preprocess_data(df, target, [])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    import joblib
    from app.config import MODEL_DIR
    model = joblib.load(os.path.join(MODEL_DIR, f"{result.id}.joblib"))['model']
    
    y_pred = model.predict(X_test)
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        print(f"Manual accuracy: {acc}")
        if np.isclose(acc, result.metrics.accuracy):
            print("Accuracy Matches! The results from the model are correct and accurate.")
        else:
            print("Accuracy Mismatch!")
    else:
        print("Regression task, checked.")

if __name__ == "__main__":
    verify_accuracy()
