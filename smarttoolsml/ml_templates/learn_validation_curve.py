import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier

sns.set_style("darkgrid")


def plot_validation_curve(X, y):
    """_summary_

    Example usage:
        # Validierungskurve kann genutzt werden um idealen Hyperparameter zu finden (Basis der Train/Testdaten)
    """

    # parameter range
    param_range = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    # train, test scores
    train_scores, test_scores = validation_curve(
        KNeighborsClassifier(),
        X,
        y,
        param_name="n_neighbors",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    # mean scores train, test
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(
        param_range,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training Accuracy",
    )
    plt.plot(
        param_range,
        test_mean,
        color="orange",
        marker="s",
        markersize=5,
        label="Validation Accuracy",
    )
    plt.xlim(np.max(param_range), np.min(param_range))
    plt.xlabel("Hyperparameter")
    plt.ylabel("Accuracy")
    plt.title("Validation Curve")
    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(X, y):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_

    Example usage:
        # die learning curve gibt an ab wievielen Trainingsbeispielen das Model overfitted
        # oder ob noch Trainingsbeispiele hinzugefügt werden können, weil es noch nicht overfitted
    """
    # calculate learning curve
    train_sizes_abs, train_scores, test_scores = learning_curve(
        KNeighborsClassifier(n_neighbors=5), X, y, cv=5, scoring="accuracy", n_jobs=-1
    )

    # calculate mean
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes_abs,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training Accuracy",
    )
    plt.plot(
        train_sizes_abs,
        test_mean,
        color="orange",
        marker="s",
        markersize=5,
        label="Validation Accuracy",
    )
    plt.xlabel("Count Train Samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()
