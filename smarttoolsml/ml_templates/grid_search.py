from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
import numpy as np


def random_search_with_pipeline(
    pipeline,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 25,
):
    """
    Conducts a random search to optimize model parameters using a provided pipeline over specified parameters,
    training data, and testing conditions. The function fits the model using the best parameters found
    and evaluates it on the test data.

    Args:
        pipeline: The pipeline object containing processing steps and the estimator. For example,
                  Pipeline([("scaler", StandardScaler()), ("classifier", SVC())]).
        params (dict): Dictionary specifying the parameters to be tested in the random search. Each key corresponds
                       to a parameter name in the pipeline, and the value is a distribution or a list of values to sample from.
        X (array-like): Feature dataset used for training and testing the model.
        y (array-like): Target variable associated with the features in X.
        test_size (float): Proportion of the dataset to be used as test set.
        random_state (int): Seed used by the random number generator for reproducible results.
        n_iter (int): Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
        cv (int): Number of folds to use for cross-validation.

    Returns:
        None. Outputs the accuracy of the model on the test set and prints the best hyperparameters found.

    Example usage:
        cv = StratifiedKFold(n_splits=5)
        steps = [("scaler", StandardScaler()), ("SVM", SVC())]
        pipeline = Pipeline(steps)
        parameters = {"SVM__C": scipy.stats.expon(scale=100), "SVM__gamma": scipy.stats.expon(scale=.1)}
        random_search_with_pipeline(pipeline, parameters, X, y, test_size=0.3, random_state=42, n_iter=100, cv=cv)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
    )
    random_search.fit(x_train, y_train)

    print("Accuracy: {}".format(random_search.score(x_test, y_test)))
    print("Tuned Model Parameters: {}".format(random_search.best_params_))


def grid_search_with_pipeline(
    pipeline,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    test_size: float = 0.2,
    random_state: str = 42,
):
    """
    Conducts a grid search to optimize model parameters using a provided pipeline over specified parameters,
    training data, and testing conditions. The function fits the model using the best parameters found
    and evaluates it on the test data.

    Args:
        pipeline: The pipeline object containing processing steps and the estimator. For example,
                  Pipeline([("scaler", StandardScaler()), ("classifier", SVC())]).
        params (dict): Dictionary specifying the parameters to be tested in the grid search. Each key corresponds
                       to a parameter name in the pipeline, and the value is a list of settings to test.
        X (array-like): Feature dataset used for training and testing the model.
        y (array-like): Target variable associated with the features in X.
        test_size (float): Proportion of the dataset to be used as test set.
        random_state (int): Seed used by the random number generator for reproducible results.
        k (int): Number of folds to use for cross-validation.

    Returns:
        None. Outputs the accuracy of the model on the test set and prints the best hyperparameters found.

    Example usage:
        cv = StratifiedKFold(n_splits=5)
        steps = [("scaler", StandardScaler()), ("SVM", SVC())]
        pipeline = Pipeline(steps)
        parameters = {"SVM__C": [1, 10, 100], "SVM__gamma": [0.1, 0.01]}
        grid_search_with_pipeline(pipeline, parameters, X, y, test_size=0.3, random_state=42, cv=cv)
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cv = GridSearchCV(pipeline, param_grid=params, cv=cv)
    cv.fit(x_train, y_train)

    print("Accuracy: {}".format(cv.score(x_test, y_test)))
    print("Tuned Model Parameters: {}".format(cv.best_params_))
