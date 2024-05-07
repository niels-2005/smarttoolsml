import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import (
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

pipelines = [
    Pipeline([("vect", CountVectorizer()), ("clf", MultinomialNB())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", SVC())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", SVC())]),
    Pipeline(
        [("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", SVC())]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", RandomForestClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", RandomForestClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RandomForestClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", GradientBoostingClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", GradientBoostingClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", GradientBoostingClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", PassiveAggressiveClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", PassiveAggressiveClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", PassiveAggressiveClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", DecisionTreeClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", DecisionTreeClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", DecisionTreeClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", RidgeClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", RidgeClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", RidgeClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", KNeighborsClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", KNeighborsClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", KNeighborsClassifier()),
        ]
    ),
    Pipeline(
        [("vect", CountVectorizer()), ("clf", AdaBoostClassifier(algorithm="SAMME"))]
    ),
    Pipeline(
        [("tfidf", TfidfVectorizer()), ("clf", AdaBoostClassifier(algorithm="SAMME"))]
    ),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", AdaBoostClassifier(algorithm="SAMME")),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", XGBClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", XGBClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", XGBClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", ExtraTreesClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", ExtraTreesClassifier())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", ExtraTreesClassifier()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", LinearSVC())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LinearSVC()),
        ]
    ),
    Pipeline([("vect", CountVectorizer()), ("clf", Perceptron())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("clf", Perceptron())]),
    Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", Perceptron()),
        ]
    ),
]


def compare_pipelines(
    X_train: np.ndarray, y_train: np.ndarray, cv, plot_comparison: bool = True
) -> None:
    """
    Compares multiple machine learning pipelines on the given training data using cross-validation and optionally plots the comparison.

    Args:
        X_train (np.ndarray): The training input samples.
        y_train (np.ndarray): The target labels for the training input samples.
        cv (object): Cross-validation splitting strategy, e.g., an instance of StratifiedKFold.
        plot_comparison (bool): If True, plot the accuracy comparison as a horizontal bar chart (default is True).

    Returns:
        None: This function does not return a value but prints out the accuracy for each classifier and may display a plot.

    Example usage:
        cv = StratifiedKFold(n_splits=5)
        X_train = X_train[:5000]
        y_train = y_train[:5000]
        compare_pipelines(X_train, y_train, cv=cv, plot_comparison=True)
    """
    pipeline_descriptions = []
    accuracy_scores = []

    for idx, pipeline in enumerate(pipelines):
        step_names = " | ".join([type(step[1]).__name__ for step in pipeline.steps])
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="accuracy"
        )
        mean_score = np.mean(cv_scores)
        pipeline_descriptions.append(step_names)
        accuracy_scores.append(mean_score)
        print(f"Pipeline {idx + 1}: {step_names}, Accuracy: {mean_score:.4f}")

    if plot_comparison:
        zipped_lists = zip(accuracy_scores, pipeline_descriptions)
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

        plt.xlabel("Accuracy")
        plt.title("Classifier Performance Comparison")
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.show()


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
