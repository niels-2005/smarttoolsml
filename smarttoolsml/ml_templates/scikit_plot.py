import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.decomposition import PCA


def plot_confusion_matrix(y_true, y_pred):
    skplt.metrics.plot_confusion_matrix(y_pred=y_pred, y_true=y_true)
    plt.show()


def plot_roc_curve(y_true, y_probas):
    skplt.metrics.plot_roc(y_true=y_true, y_probas=y_probas)
    plt.show()


def plot_precision_recall_curve(y_true, y_probas):
    skplt.metrics.plot_precision_recall(y_probas=y_probas, y_true=y_true)
    plt.show()


def plot_learning_curve(clf, X, y):
    skplt.estimators.plot_learning_curve(clf, X, y)
    plt.show()


def plot_pca_2d(X, y):
    pca = PCA()
    pca.fit(X)
    skplt.decomposition.plot_pca_2d_projection(pca, X, y)
    plt.show()
