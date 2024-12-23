import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def get_lda_components_performance(X: np.ndarray, y: np.ndarray):
    """
    Visualize LDA explained variance components in a way similar to PCA.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector of shape (n_samples,)

    Example usage:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        get_lda_components_performance_adjusted(X=X_scaled, y=y)
    """
    # Fit LDA
    lda = LDA().fit(X, y)

    eigenvalues = lda.explained_variance_ratio_
    n_comps = list(range(1, len(eigenvalues) + 1))
    cum_var_exp = np.cumsum(eigenvalues)

    # Adjust to match PCA-like visuals
    plt.figure(figsize=(10, 4))
    plt.bar(
        n_comps,
        eigenvalues,
        alpha=0.5,
        label="Individual explained variance",
    )
    plt.step(
        n_comps,
        cum_var_exp,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.xlabel("Linear Discriminant Index")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(n_comps)
    plt.title("LDA Explained Variance (PCA-like Visualization)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def lda_get_best_components(
    model,
    X,
    y,
    n_components: list,
    plot_analysis: bool = True,
    return_metrics: bool = False,
):
    """
    Analyzes the performance of a given model with varying numbers of LDA components.

    This function iteratively applies LDA to the dataset with different numbers of components specified,
    trains the model on this transformed dataset, and records performance metrics such as accuracy,
    F1 score, precision, and recall.

    Args:
        model (estimator): A machine learning model that follows the scikit-learn estimator interface.
        X (ndarray): The feature dataset which needs to be transformed by LDA.
        y (ndarray): The target labels.
        n_components (list): A list of integers where each integer specifies a number of linear
                             discriminant components to keep while transforming the dataset with LDA.
        plot_analysis (bool, optional): If True, plots the performance metrics after analysis. Defaults to True.
        return_metrics (bool, optional): If True, returns a dictionary containing performance metrics. Defaults to False.

    Example usage:
        n_components = [1, 2]

        model = linear_model.LogisticRegression(solver="lbfgs")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = lda_get_best_components(model=model, X=X_scaled, y=y, n_components=n_components)
    """
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    max_components = min(X.shape[1], len(np.unique(y)) - 1)

    for n in n_components:
        if n > max_components:
            print(
                f"Skipping n_components={n}: Maximum allowed components are {max_components}"
            )
            continue

        lda = LDA(n_components=n)
        x_trans = lda.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            x_trans, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
            y_pred=y_pred, y_true=y_test, average="weighted"
        )
        accuracy_scores.append(accuracy)
        f1_scores.append(model_f1)
        precision_scores.append(model_precision)
        recall_scores.append(model_recall)

    metrics = {
        "Components": n_components[: len(accuracy_scores)],
        "Accuracy": accuracy_scores,
        "F1 Score": f1_scores,
        "Precision Score": precision_scores,
        "Recall Score": recall_scores,
    }

    if plot_analysis:
        plot_lda_performance(metrics=metrics)

    if return_metrics:
        return metrics


def plot_lda_performance(metrics: dict) -> None:
    """
    Plots the performance metrics as a function of the number of LDA components used in the model.

    Args:
        metrics (dict): A dictionary containing the LDA components and their corresponding performance metrics
                        such as accuracy, F1 score, precision, and recall.

    The function generates a line plot for each metric with LDA components on the x-axis and the metric score on the y-axis.
    """
    df_score = pd.DataFrame(metrics)

    plt.figure(figsize=(14, 8))
    plt.plot(df_score["Components"], df_score["Accuracy"], marker="o", label="Accuracy")
    plt.plot(df_score["Components"], df_score["F1 Score"], marker="o", label="F1 Score")
    plt.plot(
        df_score["Components"],
        df_score["Precision Score"],
        marker="o",
        label="Precision Score",
    )
    plt.plot(
        df_score["Components"],
        df_score["Recall Score"],
        marker="o",
        label="Recall Score",
    )

    plt.title("Model Performance Metrics by Number of LDA Components")
    plt.xlabel("Number of LDA Components")
    plt.xticks(df_score["Components"], df_score["Components"].apply(str))
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
