# MLflow Evaluate API

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

# Load data and train model
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

with mlflow.start_run():
    # Log model with signature
    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    # Comprehensive evaluation with MLflow
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",  # or "regressor" for regression
        evaluators=["default"],
    )

    # Access automatic metrics
    print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"F1 Score: {result.metrics['f1_score']:.3f}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")

    # Access generated artifacts
    print("Generated artifacts:")
    for artifact_name, path in result.artifacts.items():
        print(f"  {artifact_name}: {path}")


# REGRESSION Evaluation

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load regression dataset
housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Create evaluation dataset
eval_data = X_test.copy()
eval_data["target"] = y_test

with mlflow.start_run():
    # Log and evaluate regression model
    signature = infer_signature(X_train, reg_model.predict(X_train))
    mlflow.sklearn.log_model(reg_model, name="model", signature=signature)
    model_uri = mlflow.get_artifact_uri("model")

    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )

    print(f"MAE: {result.metrics['mean_absolute_error']:.3f}")
    print(f"RMSE: {result.metrics['root_mean_squared_error']:.3f}")
    print(f"R² Score: {result.metrics['r2_score']:.3f}")


# Custom Metric & Artifacts

from mlflow.models import make_metric
import matplotlib.pyplot as plt
import numpy as np
import os


def business_value_metric(predictions, targets, sample_weights=None):
    """Custom business metric: value from correct predictions."""
    # Assume $50 value per correct prediction, $20 cost per error
    correct_predictions = (predictions == targets).sum()
    incorrect_predictions = len(predictions) - correct_predictions

    business_value = (correct_predictions * 50) - (incorrect_predictions * 20)
    return business_value


def create_feature_distribution_plot(eval_df, builtin_metrics, artifacts_dir):
    """Create feature distribution plots for model analysis."""

    # Select numeric features for distribution analysis
    numeric_features = eval_df.select_dtypes(include=[np.number]).columns
    numeric_features = [
        col for col in numeric_features if col not in ["label", "prediction"]
    ]

    if len(numeric_features) > 0:
        # Create subplot for feature distributions
        n_features = min(6, len(numeric_features))  # Show up to 6 features
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, feature in enumerate(numeric_features[:n_features]):
            axes[i].hist(eval_df[feature], bins=30, alpha=0.7, edgecolor="black")
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        plot_path = os.path.join(artifacts_dir, "feature_distributions.png")
        plt.savefig(plot_path)
        plt.close()

        return {"feature_distributions": plot_path}

    return {}


# Create custom metric
custom_business_value = make_metric(
    eval_fn=business_value_metric, greater_is_better=True, name="business_value_score"
)

# Use custom metrics and artifacts
result = mlflow.evaluate(
    model_uri,
    eval_data,
    targets="label",
    model_type="classifier",
    extra_metrics=[custom_business_value],
    custom_artifacts=[create_feature_distribution_plot],
)

print(f"Business Value Score: ${result.metrics['business_value_score']:.2f}")
