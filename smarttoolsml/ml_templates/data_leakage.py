import pandas as pd
from sklearn.metrics import mutual_info_score


def find_leaking_features(X_train, X_test, y_train, y_test):

    leak_scores = []

    for col in X_train.columns:

        X_train_col = X_train[[col]]
        X_test_col = X_test[[col]]

        score = mutual_info_score(X_train_col, y_train) - mutual_info_score(
            X_test_col, y_test
        )

        leak_scores.append(score)

    leaks = pd.DataFrame(
        list(zip(X_train.columns, leak_scores)), columns=["feature", "leak_score"]
    )

    return leaks
