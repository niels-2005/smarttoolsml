from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.neighbors 
import numpy as np


def simple_knn_classifier(knn: sklearn.neighbors.KNeighborsClassifier, 
                          X: np.ndarray, 
                          y: np.ndarray,
                          test_size: float = 0.3,
                          random_state: int = 42):
    """_summary_

    Args:
        knn (sklearn.neighbors.KNeighborsClassifier): _description_
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        test_size (float, optional): _description_. Defaults to 0.3.
        random_state (int, optional): _description_. Defaults to 42.

    Example usage:
        knn = KNeighborsClassifier(5)
        simple_knn_classifier(knn, X, y, test_size=0.3)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(f"With KNN accuracy is: {knn.score(X_test, y_test)}")




