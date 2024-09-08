import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score


def cv_regression(X, y, model, cv):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        model (_type_): _description_
        n_splits (int, optional): _description_. Defaults to 5.
        shuffle (bool, optional): _description_. Defaults to True.

    Example usage:
        model = LinearRegression()
        X = df["..."].values
        y = df["..."].values
        cv = KFold(n_splits=n_splits, shuffle=shuffle)

        simple_kfold_regression(X=X, y=y, model=model, cv=cv)
    """

    for train_index, test_index in cv.split(X):
        # correct splits
        X_test = X[test_index]
        X_train = X[train_index]
        y_test = y[test_index]
        y_train = y[train_index]

        # fit model
        model.fit(X_train, y_train)

        # print model score
        print(model.score(X_test, y_test))


def cross_val_score_regression(X, y, model, cv):
    """_summary_

    Args:
        X (_type_): _description_
        y (_type_): _description_
        model (_type_): _description_
        cv (_type_): _description_

    Example usage:
        model = LinearRegression()
        X = df["..."].values
        y = df["..."].values
        cv = RepeatedKFold(n_repeats=1000)

        cross_val_score_regression(X=X, y=y, model=model, cv=cv)
    """
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    print(np.mean(scores))
