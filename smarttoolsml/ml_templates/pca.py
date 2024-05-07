import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_pca_components_performance(X: np.ndarray):
    """_summary_

    Args:
        X (np.ndarray): _description_
    
    Example usage:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        get_pca_components_performance(X=X_scaled)
    """
    pca = PCA(whiten=True).fit(X)
    var_exp = pca.explained_variance_ratio_
    n_comps=list(range(1, len(var_exp)+1))

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(n_comps, var_exp)
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance by Number of Components")

    plt.subplot(1, 2, 2)
    plt.plot(n_comps, np.cumsum(var_exp))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance by Number of Components")
    plt.show()


def pca_get_best_components(model, X, y, n_components: list, plot_analysis: bool = True, return_metrics: bool = False):
    """
    Analyzes the performance of a given model with varying numbers of PCA components.

    This function iteratively applies PCA to the dataset with different numbers of components specified,
    trains the model on this transformed dataset, and records performance metrics such as accuracy,
    F1 score, precision, and recall.

    Args:
        model (estimator): A machine learning model that follows the scikit-learn estimator interface.
        X (ndarray): The feature dataset which needs to be transformed by PCA.
        y (ndarray): The target labels.
        n_components (list): A list of integers where each integer specifies a number of principal
                             components to keep while transforming the dataset with PCA.
        plot_analysis (bool, optional): If True, plots the performance metrics after analysis. Defaults to True.
        return_metrics (bool, optional): If True, returns a dictionary containing performance metrics. Defaults to False.

    Example usage:
        n_components = [25, 50, 75, 100, 125, 150, 200, 250, 300]
        
        model = linear_model.LogisticRegression(solver="lbfgs")
        
        scaler = StandardScaler() # or
        scaler = RobustScaler() # or
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        results = pca_get_best_components(model=model, X=X_scaled, y=y, n_components=n_components)
    """
    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for n in n_components:
        pca = PCA(n_components=n, whiten=True)
        x_trans = pca.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(x_trans, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average="weighted")
        accuracy_scores.append(accuracy)
        f1_scores.append(model_f1)
        precision_scores.append(model_precision)
        recall_scores.append(model_recall)

    metrics = {
        "Components": n_components,
        "Accuracy": accuracy_scores,
        "F1 Score": f1_scores,
        "Precision Score": precision_scores,
        "Recall Score": recall_scores
    }
    if plot_analysis:
        plot_pca_performance(metrics=metrics)
    
    if return_metrics:
        return metrics


def plot_pca_performance(metrics: dict) -> None:
    """
    Plots the performance metrics as a function of the number of PCA components used in the model.

    Args:
        metrics (dict): A dictionary containing the PCA components and their corresponding performance metrics
                        such as accuracy, F1 score, precision, and recall.

    The function generates a line plot for each metric with PCA components on the x-axis and the metric score on the y-axis.
    """
    df_score = pd.DataFrame(metrics)

    plt.figure(figsize=(14, 8))
    plt.plot(df_score['Components'], df_score['Accuracy'], marker='o', label='Accuracy')
    plt.plot(df_score['Components'], df_score['F1 Score'], marker='o', label='F1 Score')
    plt.plot(df_score['Components'], df_score['Precision Score'], marker='o', label='Precision Score')
    plt.plot(df_score['Components'], df_score['Recall Score'], marker='o', label='Recall Score')

    plt.title('Model Performance Metrics by Number of PCA Components')
    plt.xlabel('Number of PCA Components')
    plt.xticks(df_score['Components'], df_score['Components'].apply(str))
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.show()