import numpy as np
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

pipelines = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", MultiOutputClassifier(KNeighborsClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", RandomForestClassifier(n_jobs=-1))]),
    Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", RobustScaler()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("scaler", StandardScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline([("scaler", MinMaxScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline([("scaler", RobustScaler()), ("clf", ExtraTreesClassifier(n_jobs=-1))]),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(BaggingClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", MultiOutputClassifier(BaggingClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline(
        [
            ("scaler", RobustScaler()),
            ("clf", MultiOutputClassifier(BaggingClassifier(n_jobs=-1))),
        ]
    ),
    Pipeline(
        [("scaler", StandardScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))]
    ),
    Pipeline(
        [("scaler", MinMaxScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))]
    ),
    Pipeline(
        [("scaler", RobustScaler()), ("clf", MultiOutputClassifier(MLPClassifier()))]
    ),
]


def get_probabilites(
    pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
):
    """_summary_

    Args:
        pipeline (_type_): _description_
        X_train (np.ndarray): _description_
        y_train (np.ndarray): _description_
        X_test (np.ndarray): _description_

    Returns:
        _type_: _description_

    Example usage:
        pipeline = Pipeline([("scaler", RobustScaler()), ("clf", RandomForestClassifier())])
        X_train = X_train[:7500]
        y_train = y_train[:7500]
        X_test = X_test[:7500]
        probs = get_probabilites(pipeline=pipeline, X_train, y_train, X_test)
    """
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    return probs
