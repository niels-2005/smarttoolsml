import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd


def evaluate_cross_val_scores(model, X, y, cv):
    """_summary_

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_
        cv (_type_): _description_

    Example usage:
        clf = LogisticRegression()
        cv = StratifiedKFold(n_splits=5)
        evaluate_cross_val_scores(model=clf, X=X, y=y, cv=cv)
    """
    scores = ["accuracy", "precision", "recall", "f1"]
    mean_scores = {}
    for score in scores:
        result = cross_val_score(model, X, y, cv=cv, scoring=score)
        mean_scores[score] = np.mean(result)

    df = pd.DataFrame(mean_scores, index=[0])
    ax = df.T.plot(kind="barh", legend=False)

    for i, v in enumerate(df.values.flatten()):
        ax.text(v + 0.01, i - 0.1, f"{v:.2f}", color="black")

    plt.title(f"Metrics with Cross Validation")
    plt.show()
