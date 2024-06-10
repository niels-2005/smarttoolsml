import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.linear_model import (PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pipelines = [
    Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]),
    Pipeline([("vect", CountVectorizer()), ("clf", BernoulliNB())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", BernoulliNB())]),
    Pipeline([("vect", CountVectorizer()), ("clf", SVC())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", SVC())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", RandomForestClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", GradientBoostingClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", PassiveAggressiveClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", PassiveAggressiveClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", DecisionTreeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", DecisionTreeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline(
        [("vect", CountVectorizer()), ("clf", RidgeClassifier(class_weight="balanced"))]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
    Pipeline(
        [("vect", CountVectorizer()), ("clf", AdaBoostClassifier(algorithm="SAMME"))]
    ),
    Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", AdaBoostClassifier(algorithm="SAMME"))]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", XGBClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", XGBClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", ExtraTreesClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", ExtraTreesClassifier(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [("vect", CountVectorizer()), ("clf", LinearSVC(class_weight="balanced"))]
    ),
    Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", LinearSVC(class_weight="balanced"))]
    ),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("clf", Perceptron(n_jobs=-1, class_weight="balanced")),
        ]
    ),
    Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", Perceptron(n_jobs=-1, class_weight="balanced")),
        ]
    ),
]


def compare_pipelines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    metric: str = "accuracy",
    plot_comparison: bool = True,
    return_df: bool = True,
) -> pd.DataFrame:
    """
    Compares multiple machine learning pipelines on the given training data using cross-validation and optionally plots the comparison.
    Optionally returns a DataFrame containing the results.

    Args:
        X_train (np.ndarray): The training input samples.
        y_train (np.ndarray): The target labels for the training input samples.
        cv (object): Cross-validation splitting strategy, e.g., an instance of StratifiedKFold.
        metric (str): Scoring metric to evaluate the models.
        plot_comparison (bool): If True, plot the metric comparison as a horizontal bar chart.
        return_df (bool): If True, returns a DataFrame containing the evaluation results.

    Returns:
        pd.DataFrame or None: Returns a DataFrame with the pipeline descriptions, metrics, and scores if return_df is True, otherwise returns None.

    Scoring Metric Details:
        'accuracy': Use when each class prediction is equally important.
        'f1_micro': Use when you want to weight each instance or prediction equally. Ideal for imbalanced datasets.
        'f1_macro': Use when you want to treat all classes equally regardless of their frequency. Good for when class distribution is not representative of importance.
        'f1_weighted': Use when class imbalance might affect the model's performance, and you want to weight the metric towards the more frequent classes.
        'precision_micro': Similar to 'f1_micro', focus on the total proportion of true positive identifications across all classes.
        'precision_macro': Similar to 'f1_macro', calculates metrics independently for each class but does not take class imbalance into account.
        'precision_weighted': Useful for taking the relative frequency of each class into account, thereby weighting the precision of each class by its presence in the dataset.
        'recall_micro': Focuses on the overall ability of the model to correctly identify all labels across all classes.
        'recall_macro': Evaluates the model’s ability to detect each class, treating all classes as equally important.
        'recall_weighted': Adjusts recall for the frequency of each class, providing a metric that reflects the relative importance of classes.

    Example usage:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_train = X_train[:5000]
        y_train = y_train[:5000]

        metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

        for metric in metrics:
            df = compare_pipelines(X_train, y_train, cv=cv, metric=metric, plot_comparison=True, return_df=True)
    """
    pipeline_descriptions = []
    scores = []

    for idx, pipeline in enumerate(pipelines):
        step_names = " | ".join([type(step[1]).__name__ for step in pipeline.steps])
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=metric
        ).mean()
        pipeline_descriptions.append(step_names)
        scores.append(cv_scores)
        print(f"Pipeline {idx + 1}: {step_names}, {metric}: {cv_scores:.4f}")

    if plot_comparison:
        zipped_lists = zip(scores, pipeline_descriptions)
        sorted_pairs = sorted(zipped_lists, reverse=True, key=lambda x: x[0])
        sorted_scores, sorted_names = zip(*sorted_pairs)

        plt.figure(figsize=(10, 8))
        bars = plt.barh(sorted_names, sorted_scores, color="skyblue")
        for bar in bars:
            plt.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}",
                va="center",
            )
        plt.xlabel(f"{metric.capitalize()}")
        plt.title("Classifier Performance Comparison")
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.show()

    if return_df:
        results_df = pd.DataFrame(
            {"Pipeline": pipeline_descriptions, "Metric": metric, "Score": scores}
        )
        return results_df
    return None


def predict_sample(text: str, pipeline):
    """
    Predicts the class of a given text using a specified machine learning pipeline.

    Args:
        text (str): The text sample to be classified.
        pipeline: The machine learning pipeline (fitted) to be used for prediction.

    Returns:
        None: This function does not return a value but prints out the predicted label for the given text.

    Example usage:
        text = "Sample text that needs classification."
        ada_pipeline = Pipeline([('vect', CountVectorizer()), ('clf', AdaBoostClassifier(algorithm='SAMME'))])
        predict_sample(text, ada_pipeline)
    """
    predicted_class = pipeline.predict([text])
    print(f"Text:\n{text}\n\nPredicted Label: {predicted_class}")
