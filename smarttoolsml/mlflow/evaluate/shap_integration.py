import mlflow
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

# Load the UCI Adult Dataset
X, y = shap.datasets.adult()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train model
model = xgb.XGBClassifier().fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data["label"] = y_test

with mlflow.start_run():
    # Log model
    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    # Evaluate with SHAP explanations enabled
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
        evaluator_config={"log_explainer": True},  # Enable SHAP logging
    )

    print("SHAP artifacts generated:")
    for artifact_name in result.artifacts:
        if "shap" in artifact_name.lower():
            print(f"  - {artifact_name}")


# UNDERSTANDING SHAP OUTPUTS

# Access the results
print(f"Model accuracy: {result.metrics['accuracy_score']:.3f}")
print("Generated SHAP artifacts:")
for name, path in result.artifacts.items():
    if "shap" in name:
        print(f"  {name}: {path}")


# CONFIGURING SHAP

# Advanced SHAP configuration
shap_config = {
    "log_explainer": True,  # Save the explainer model
    "explainer_type": "exact",  # Use exact SHAP values (slower but precise)
    "max_error_examples": 100,  # Number of error cases to explain
    "log_model_explanations": True,  # Log individual prediction explanations
}

result = mlflow.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    evaluators=["default"],
    evaluator_config=shap_config,
)

# more in mlflow doc
