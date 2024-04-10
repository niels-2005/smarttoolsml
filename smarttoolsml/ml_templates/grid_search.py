from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def grid_search_with_pipeline(pipeline, params, X, y, test_size, random_state, k):
    """_summary_

    Example usage:
        steps = [("scalar", StandardScaler()), ("SVM", SVC())]
        pipeline = Pipeline(steps)
        parameters = {"SVM__C": [1, 10, 100], "SVM__gamma": [0.1, 0.01]}

        grid_search_with_pipeline(pipeline, params, X, y, test_size=0.3)
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    cv = GridSearchCV(pipeline, param_grid=params, cv=k)
    cv.fit(x_train, y_train)
    y_pred = cv.predict(x_test)

    print("Accuracy: {}".format(cv.score(x_test, y_test)))
    print("Tuned Model Parameters: {}".format(cv.best_params_))


    

