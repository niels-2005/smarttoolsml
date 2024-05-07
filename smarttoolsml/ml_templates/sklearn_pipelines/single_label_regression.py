import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

pipelines = [
    Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", LinearRegression())]),
    Pipeline([("scaler", RobustScaler()), ("reg", LinearRegression())]),
    Pipeline([("scaler", StandardScaler()), ("reg", Ridge())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", Ridge())]),
    Pipeline([("scaler", RobustScaler()), ("reg", Ridge())]),
    Pipeline([("scaler", StandardScaler()), ("reg", Lasso())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", Lasso())]),
    Pipeline([("scaler", RobustScaler()), ("reg", Lasso())]),
    Pipeline([("scaler", StandardScaler()), ("reg", ElasticNet())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", ElasticNet())]),
    Pipeline([("scaler", RobustScaler()), ("reg", ElasticNet())]),
    Pipeline([("scaler", StandardScaler()), ("reg", SVR())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", SVR())]),
    Pipeline([("scaler", RobustScaler()), ("reg", SVR())]),
    Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", RandomForestRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", RandomForestRegressor())]),
    Pipeline([("scaler", StandardScaler()), ("reg", GradientBoostingRegressor())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", GradientBoostingRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", GradientBoostingRegressor())]),
    Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", KNeighborsRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", KNeighborsRegressor())]),
    Pipeline([("scaler", StandardScaler()), ("reg", DecisionTreeRegressor())]),
    Pipeline([("scaler", MinMaxScaler()), ("reg", DecisionTreeRegressor())]),
    Pipeline([("scaler", RobustScaler()), ("reg", DecisionTreeRegressor())]),
]


def compare_pipelines(
    X_train: np.ndarray, y_train: np.ndarray, cv, plot_comparison: bool = True
) -> None:
    """
    Compares multiple machine learning pipelines on the given training data using cross-validation and optionally plots the comparison for MSE, MAE, and R².

    Args:
        X_train (np.ndarray): The training input samples.
        y_train (np.ndarray): The target labels for the training input samples.
        cv (object): Cross-validation splitting strategy.
        plot_comparison (bool): If True, plot the MSE, MAE, and R² comparison as a horizontal bar chart.

    Returns:
        None: This function does not return a value but prints out the metrics for each regressor and may display a plot.

    Example usage:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5)
        compare_pipelines(X_train, y_train, cv=cv, plot_comparison=True)
    """
    metrics = {
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "R2": make_scorer(r2_score),
    }

    results = {metric: [] for metric in metrics}
    descriptions = []

    for idx, pipeline in enumerate(pipelines):
        step_names = " | ".join([type(step[1]).__name__ for step in pipeline.steps])
        descriptions.append(step_names)

        for metric, scorer in metrics.items():
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
            results[metric].append(np.mean(scores))

        print(f"Pipeline {idx + 1}: {step_names}")
        for metric in metrics:
            print(f"  {metric}: {results[metric][-1]:.4f}")

    if plot_comparison:
        plt.figure(figsize=(15, 10))
        num_metrics = len(metrics)
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, num_metrics, i)
            sorted_indices = np.argsort(results[metric])
            sorted_scores = [results[metric][idx] for idx in sorted_indices]
            sorted_names = [descriptions[idx] for idx in sorted_indices]

            bars = plt.barh(sorted_names, sorted_scores, color="skyblue")
            for bar in bars:
                plt.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va="center",
                )

            plt.xlabel(metric)
            plt.title(f"{metric} Comparison")

        plt.tight_layout()
        plt.show()
