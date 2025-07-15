# GRIDSEARCH CV

import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Enable autologging with hyperparameter tuning support
mlflow.sklearn.autolog(max_tuning_runs=10)  # Track top 10 parameter combinations

# Load data
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

with mlflow.start_run(run_name="Random Forest Hyperparameter Tuning"):
    # Create and fit GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best model evaluation
    best_score = grid_search.score(X_test, y_test)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    print(f"Test score: {best_score:.3f}")


# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions for more efficient exploration
param_distributions = {
    "n_estimators": randint(50, 300),
    "max_depth": randint(5, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),
}

with mlflow.start_run(run_name="Randomized Hyperparameter Search"):
    rf = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=50,  # Try 50 random combinations
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    # MLflow automatically creates child runs for parameter combinations
    # The parent run contains the best model and overall results
