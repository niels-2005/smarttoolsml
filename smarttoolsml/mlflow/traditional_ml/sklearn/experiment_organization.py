import mlflow
from sklearn.ensemble import RandomForestClassifier

X_train, y_train = None, None

# Organize experiments with descriptive names and tags
experiment_name = "Customer Churn Prediction - Q4 2024"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="Baseline Random Forest"):
    # Use consistent tagging for easy filtering and organization
    mlflow.set_tags(
        {
            "model_type": "ensemble",
            "algorithm": "random_forest",
            "dataset_version": "v2.1",
            "feature_engineering": "standard",
            "purpose": "baseline",
        }
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
