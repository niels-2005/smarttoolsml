from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
import numpy as np
import matplotlib.pyplot as plt

def get_k_best_features(k: int , X: np.ndarray, y: np.ndarray, transform_data: bool = False):
    """_summary_

    Args:
        k (int): _description_
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        transform_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    selected_features = SelectKBest(chi2, k=k).fit(X, y)
    print("Score list:", selected_features.scores_)
    print("Feature list:", X.columns)

    if transform_data:
        X = selected_features.transform(X)
        y = selected_features.transform(y)
        return X, y


def rfecv_feature_selector(model, 
                           X: np.ndarray, 
                           y: np.ndarray, 
                           step: int = 1, 
                           cv: int = 5, 
                           scoring: str = "accuracy",
                           plot_curve: bool = True):
    """_summary_

    Args:
        model (_type_): _description_
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        step (int, optional): _description_. Defaults to 1.
        cv (int, optional): _description_. Defaults to 5.
        scoring (str, optional): _description_. Defaults to "accuracy".

    Example usage:
        rfc = RandomForestClassifier()
        rfecv = rfecv_feature_selector(model=rfc, X, y, cv=5, plot_curve=True)
    """
    rfecv = RFECV(
        estimator=model, step=step, cv=cv, scoring=scoring
    )
    rfecv.fit(X, y)
    print("Optimal number of features :", rfecv.n_features_)
    print("Best features :", X.columns[rfecv.support_])

    if plot_curve:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (mean accuracy) of number of selected features")
        plt.plot(
            range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
            rfecv.cv_results_["mean_test_score"],
        )
        plt.show()

    return rfecv
