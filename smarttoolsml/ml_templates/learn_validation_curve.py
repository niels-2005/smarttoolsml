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

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(
        param_range,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )

    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        param_range,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )

    plt.fill_between(
        param_range,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.legend(loc="lower right")
    plt.xlim(np.max(param_range), np.min(param_range))
    plt.xlabel("Hyperparameter")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    # plt.savefig('images/06_06.png', dpi=300)
    plt.show()


def plot_learning_curve(X, y, clf):
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
        clf, X, y, cv=10, scoring="accuracy", n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        train_sizes_abs,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training accuracy",
    )

    plt.fill_between(
        train_sizes_abs,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes_abs,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="Validation accuracy",
    )

    plt.fill_between(
        train_sizes_abs,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0.8, 1.03])
    plt.tight_layout()
    # plt.savefig('images/06_05.png', dpi=300)
    plt.show()
