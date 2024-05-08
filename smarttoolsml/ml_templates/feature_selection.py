import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV, SelectKBest, chi2


def get_k_best_features(
    k: int, X: np.ndarray, y: np.ndarray, transform_data: bool = False
):
    """_summary_

    Args:
        k (int): _description_
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        transform_data (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    Example usage:
        get_k_best_features(k=5, X=X_norm, y=y_train)

    """
    selected_features = SelectKBest(chi2, k=k).fit(X, y)
    print("Score list:", selected_features.scores_)
    print("Feature list:", X.columns)

    if transform_data:
        X = selected_features.transform(X)
        y = selected_features.transform(y)
        return X, y


def rfecv_feature_selector(
    model,
    X: np.ndarray,
    y: np.ndarray,
    step: int = 1,
    cv: int = 5,
    scoring: str = "accuracy",
    plot_curve: bool = True,
):
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
        cv = StratifiedKFold(n_splits=5)
        rfecv = rfecv_feature_selector(model=rfc, X, y, cv=cv, plot_curve=True)
    """
    rfecv = RFECV(estimator=model, step=step, cv=cv, scoring=scoring, n_jobs=-1)
    rfecv.fit(X, y)
    print("Optimal number of features :", rfecv.n_features_)
    print("Best features :", X.columns[rfecv.support_])

    if plot_curve:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel(
            "Cross validation score (mean accuracy) of number of selected features"
        )
        plt.plot(
            range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
            rfecv.cv_results_["mean_test_score"],
        )
        plt.show()

    return rfecv


def cor_selector(X: np.ndarray, y: np.ndarray, num_feats: int):
    """_summary_

    Args:
        X (np.ndarray): _description_
        y (np.ndarray): _description_
        num_feats (int): _description_

    Returns:
        _type_: _description_

    Example usage:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        (str(len(cor_feature)), 'selected features')

    """
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
