import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score)


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


def cross_val_score_api(X, y, model, cv):
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
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring="accuracy")
    print(np.mean(scores))


def stratified_k_fold(X_train, y_train, clf):
    kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)

    scores = []

    for k, (train, test) in enumerate(kfold):
        clf.fit(X_train[train], y_train[train])

        score = clf.score(X_train[test], y_train[test])

        scores.append(score)

        print(f"Fold: {k+1}, Class dist.: {np.bincount(y_train[train])}, Acc: {score}")

    print(f"KV-Classification: {np.mean(scores)} +/- {np.std(scores)}")


def cross_validation_5x2(clf, X_train, y_train, param_grid):
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring="accuracy", cv=2)

    scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
    print("CV accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))
