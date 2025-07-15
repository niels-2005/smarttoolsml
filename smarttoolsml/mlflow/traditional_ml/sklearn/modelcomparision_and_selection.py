# MLFlow Model Comparison
import mlflow
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data = pd.DataFrame(eval_data, columns=wine.feature_names)
eval_data["label"] = y_test
# Define models to compare
sklearn_models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
    "svm": SVC(probability=True, random_state=42),
}

# Evaluate each model systematically
comparison_results = {}

for model_name, model in sklearn_models.items():
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        # Train model
        model.fit(X_train, y_train)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # Comprehensive evaluation with MLflow
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
        )

        comparison_results[model_name] = result.metrics

        # Log key metrics for comparison
        mlflow.log_metrics(
            {
                "accuracy": result.metrics["accuracy_score"],
                "f1": result.metrics["f1_score"],
                "roc_auc": result.metrics["roc_auc"],
                "precision": result.metrics["precision_score"],
                "recall": result.metrics["recall_score"],
            }
        )

# Create comparison summary
import pandas as pd

comparison_df = pd.DataFrame(comparison_results).T
print("Model Comparison Summary:")
print(comparison_df[["accuracy_score", "f1_score", "roc_auc"]].round(3))

# Identify best model
best_model = comparison_df["f1_score"].idxmax()
print(f"\nBest model by F1 score: {best_model}")


# Hyperparameter Evaluation Selection
from sklearn.model_selection import ParameterGrid

# Define parameter grid for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# Evaluate each parameter combination
grid_results = []

for params in ParameterGrid(param_grid):
    with mlflow.start_run(run_name=f"rf_grid_search"):
        # Log parameters
        mlflow.log_params(params)

        # Train model with current parameters
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # Log and evaluate
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # MLflow evaluation
        result = mlflow.evaluate(
            model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
        )

        # Track results
        grid_results.append(
            {
                **params,
                "f1_score": result.metrics["f1_score"],
                "roc_auc": result.metrics["roc_auc"],
                "accuracy": result.metrics["accuracy_score"],
            }
        )

        # Log selection metric
        mlflow.log_metric("grid_search_score", result.metrics["f1_score"])

# Find best parameters
best_result = max(grid_results, key=lambda x: x["f1_score"])
print(f"Best parameters: {best_result}")
