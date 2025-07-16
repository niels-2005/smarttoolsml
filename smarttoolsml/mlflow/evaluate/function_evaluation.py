import mlflow
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a model (we'll use this in our function)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Define a prediction function
def predict_function(input_data):
    """Custom prediction function that can include business logic."""

    # Get base model predictions
    base_predictions = model.predict(input_data)

    # Add custom business logic
    # Example: Override predictions for specific conditions
    feature_sum = input_data.sum(axis=1)
    high_feature_mask = feature_sum > feature_sum.quantile(0.9)

    # Custom rule: high feature sum values are always class 1
    final_predictions = base_predictions.copy()
    final_predictions[high_feature_mask] = 1

    return final_predictions


# Create evaluation dataset
eval_data = pd.DataFrame(X_test)
eval_data["target"] = y_test

with mlflow.start_run():
    # Evaluate function directly - no model logging needed!
    result = mlflow.evaluate(
        predict_function,  # Function to evaluate here !!!
        eval_data,  # Evaluation data
        targets="target",  # Target column
        model_type="classifier",  # Task type
    )

    print(f"Function Accuracy: {result.metrics['accuracy_score']:.3f}")
    print(f"Function F1 Score: {result.metrics['f1_score']:.3f}")


# more at https://mlflow.org/docs/latest/ml/evaluation/function-eval/
