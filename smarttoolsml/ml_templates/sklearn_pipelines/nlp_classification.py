from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

pipelines = [
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())]),
    Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC())]),
    Pipeline([('vect', CountVectorizer()), ('clf', RandomForestClassifier())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', GradientBoostingClassifier())]),
    Pipeline([('vect', CountVectorizer()), ('clf', PassiveAggressiveClassifier())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', DecisionTreeClassifier())]),
    Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RidgeClassifier())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', KNeighborsClassifier())]),
    Pipeline([('vect', CountVectorizer()), ('clf', AdaBoostClassifier(algorithm='SAMME'))]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', XGBClassifier())]),
    Pipeline([('vect', CountVectorizer()), ('clf', ExtraTreesClassifier())]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', LGBMClassifier(force_col_wise=True))]),
    Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LinearSVC(dual=True))]),
    Pipeline([('tfidf', TfidfVectorizer()), ('clf', Perceptron())])
]


def compare_pipelines(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, plot_comparison: bool = True):
    """
    Compares multiple machine learning pipelines on the given training data using cross-validation and optionally plots the comparison.

    Args:
        X_train (np.ndarray): The training input samples.
        y_train (np.ndarray): The target labels for the training input samples.
        cv (int): The number of cross-validation folds to use (default is 5).
        plot_comparison (bool): If True, plot the accuracy comparison as a horizontal bar chart (default is True).

    Returns:
        None: This function does not return a value but prints out the accuracy for each classifier and may display a plot.

    Example usage:
        X_train = np.array([...])  # Training data
        y_train = np.array([...])  # Labels
        compare_pipelines(X_train, y_train, cv=5, plot_comparison=True)
    """
    classifier_names = []
    accuracy_scores = []

    for idx, pipeline in enumerate(pipelines):
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        mean_score = np.mean(cv_scores)
        classifier_name = type(pipeline.named_steps['clf']).__name__
        classifier_names.append(classifier_name)
        accuracy_scores.append(mean_score)
        print(f"Pipeline {idx + 1}: {classifier_name}, Accuracy: {mean_score:.4f}")
    
    if plot_comparison:
        zipped_lists = zip(accuracy_scores, classifier_names)
        sorted_pairs = sorted(zipped_lists, reverse=True, key=lambda x: x[0])
        sorted_scores, sorted_names = zip(*sorted_pairs)

        plt.figure(figsize=(10, 8))
        bars = plt.barh(sorted_names, sorted_scores, color='skyblue')

        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.4f}', 
                    va='center')

        plt.xlabel('Accuracy')
        plt.title('Classifier Performance Comparison')
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
    sample = [text]
    predicted_class = pipeline.predict(sample)
    print(f"Text:\n{sample}\n\nPredicted Label: {predicted_class}")
