import mlflow
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import pandas as pd

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
    # Log model with signature
    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    # Comprehensive evaluation
    result = mlflow.models.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",  # or regressor
        evaluators=["default"],
        evaluator_config={"log_explainer": True},  # Enable Shap logging
    )

    print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")

# Automatic Classification Metrics:d

# Accuracy, Precision, Recall, F1-Score
# ROC-AUC and Precision-Recall AUC
# Log Loss and Brier Score
# Confusion Matrix and Classification Report

# Automatic Regression Metrics:

# Mean Absolute Error (MAE)
# Mean Squared Error (MSE) and Root MSE
# R² Score and Adjusted R²
# Mean Absolute Percentage Error (MAPE)
# Residual plots and distribution analysis


# CUSTOM METRICS and Artifacts

import mlflow
import numpy as np
from mlflow.models import make_metric


def weighted_accuracy(predictions, targets, metrics, sample_weights=None):
    """Custom weighted accuracy metric."""
    if sample_weights is None:
        return (predictions == targets).mean()
    else:
        correct = predictions == targets
        return np.average(correct, weights=sample_weights)


# Create custom metric
custom_accuracy = make_metric(
    eval_fn=weighted_accuracy, greater_is_better=True, name="weighted_accuracy"
)

# Use in evaluation
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    extra_metrics=[custom_accuracy],
)

# OR ARTIFACTS

import matplotlib.pyplot as plt
import os


def create_residual_plot(eval_df, builtin_metrics, artifacts_dir):
    """Create custom residual plot for regression models."""

    residuals = eval_df["target"] - eval_df["prediction"]

    plt.figure(figsize=(10, 6))
    plt.scatter(eval_df["prediction"], residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plot_path = os.path.join(artifacts_dir, "residual_plot.png")
    plt.savefig(plot_path)
    plt.close()

    return {"residual_plot": plot_path}


# Use custom artifact
result = mlflow.models.evaluate(
    model_uri,
    eval_data,
    targets="target",
    model_type="regressor",
    custom_artifacts=[create_residual_plot],
)

# Working with evaluation results

# Run evaluation
result = mlflow.models.evaluate(
    model_uri, eval_data, targets="label", model_type="classifier"
)

# Access metrics
print("All Metrics:")
for metric_name, value in result.metrics.items():
    print(f"  {metric_name}: {value}")

# Access artifacts (plots, tables, etc.)
print("\nGenerated Artifacts:")
for artifact_name, path in result.artifacts.items():
    print(f"  {artifact_name}: {path}")

# Access evaluation dataset
eval_table = result.tables["eval_results_table"]
print(f"\nEvaluation table shape: {eval_table.shape}")
print(f"Columns: {list(eval_table.columns)}")


# MODEL COMPARISON

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define models to compare
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}

# Evaluate each model
results = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        # Train model
        model.fit(X_train, y_train)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # Evaluate model
        result = mlflow.models.evaluate(
            model_uri, eval_data, targets="label", model_type="classifier"
        )

        results[model_name] = result.metrics

        # Log comparison metrics
        mlflow.log_metrics(
            {
                "accuracy": result.metrics["accuracy_score"],
                "f1": result.metrics["f1_score"],
                "roc_auc": result.metrics["roc_auc"],
            }
        )

# Compare results
comparison_df = pd.DataFrame(results).T
print("Model Comparison:")
print(comparison_df[["accuracy_score", "f1_score", "roc_auc"]].round(3))


# CROSS VALIDATION

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define models to compare
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "logistic_regression": LogisticRegression(random_state=42),
    "svm": SVC(probability=True, random_state=42),
}

# Evaluate each model
results = {}

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"eval_{model_name}"):
        # Train model
        model.fit(X_train, y_train)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        model_uri = mlflow.get_artifact_uri("model")

        # Evaluate model
        result = mlflow.models.evaluate(
            model_uri, eval_data, targets="label", model_type="classifier"
        )

        results[model_name] = result.metrics

        # Log comparison metrics
        mlflow.log_metrics(
            {
                "accuracy": result.metrics["accuracy_score"],
                "f1": result.metrics["f1_score"],
                "roc_auc": result.metrics["roc_auc"],
            }
        )

# Compare results
comparison_df = pd.DataFrame(results).T
print("Model Comparison:")
print(comparison_df[["accuracy_score", "f1_score", "roc_auc"]].round(3))


# AUTOMATED SELECTION


def evaluate_and_select_best_model(
    models, X_train, y_train, eval_data, metric="f1_score"
):
    """Evaluate multiple models and select the best performer."""

    results = {}
    best_score = -1
    best_model_name = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"candidate_{model_name}"):
            # Train and evaluate
            model.fit(X_train, y_train)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, name="model", signature=signature)
            model_uri = mlflow.get_artifact_uri("model")

            result = mlflow.models.evaluate(
                model_uri, eval_data, targets="label", model_type="classifier"
            )

            score = result.metrics[metric]
            results[model_name] = score

            # Track best model
            if score > best_score:
                best_score = score
                best_model_name = model_name

            # Log selection metrics
            mlflow.log_metrics(
                {"selection_score": score, "is_best": score == best_score}
            )

    print(f"Best model: {best_model_name} (Score: {best_score:.3f})")
    return best_model_name, results


# Use automated selection
best_model, all_scores = evaluate_and_select_best_model(
    models, X_train, y_train, eval_data, metric="f1_score"
)

# MODEL VALIDATION AND QUALITY GATES

from mlflow.models import MetricThreshold

# Evaluate your model first
result = mlflow.models.evaluate(
    model_uri, eval_data, targets="label", model_type="classifier"
)

# Define static performance thresholds
static_thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.85, greater_is_better=True  # Must achieve 85% accuracy
    ),
    "precision_score": MetricThreshold(
        threshold=0.80, greater_is_better=True  # Must achieve 80% precision
    ),
    "recall_score": MetricThreshold(
        threshold=0.75, greater_is_better=True  # Must achieve 75% recall
    ),
}

# Validate against static thresholds
try:
    mlflow.validate_evaluation_results(
        candidate_result=result,
        baseline_result=None,  # No baseline comparison
        validation_thresholds=static_thresholds,
    )
    print("✅ Model meets all static performance thresholds.")
except mlflow.exceptions.ModelValidationFailedException as e:
    print(f"❌ Model failed static validation: {e}")
