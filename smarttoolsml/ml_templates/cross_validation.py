from sklearn.model_selection import cross_val_score
import numpy as np

def cross_validation(model, k: int, X: np.ndarray, y: np.ndarray):
    """_summary_

    Args:
        model (_type_): _description_
        k (int): _description_
        X (np.ndarray): _description_
        y (np.ndarray): _description_
    """
    cv_result = cross_val_score(model, X, y, cv=k)
    print("CV Scores: ", cv_result)
    print("CV scores average: ", np.sum(cv_result) / k)


