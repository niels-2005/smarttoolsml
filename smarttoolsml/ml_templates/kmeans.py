import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def get_best_cluster(
    data: pd.DataFrame,
    n_iter: int = 10,
    init: str = "k-means++",
    random_state: int = 42,
    figsize: tuple[int, int] = (10, 7),
) -> None:
    """Ellbogen Effekt beachten!!! Wo Graph abknickt sind "geeignete" Cluster

    Args:
        data (pd.DataFrame): _description_
        n_iter (int, optional): _description_. Defaults to 10.
        init (str, optional): _description_. Defaults to "k-means++".
        random_state (int, optional): _description_. Defaults to 42.
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 7).

    Example usage:
        get_best_cluster(data=df, n_iter=15)
    """
    wcss = []
    for i in range(1, n_iter):
        kmeans = KMeans(n_clusters=i, init=init, random_state=random_state).fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=figsize)
    plt.plot(range(1, n_iter), wcss)
    plt.show()


def kmeans_with_pairplot(
    data: pd.DataFrame,
    n_cluster: int,
    data_cols: list,
    init: str = "k-means++",
    random_state: int = 42,
    height: float = 1.5,
    aspect: float = 1.5,
) -> None:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        n_cluster (int): _description_
        data_cols (list): _description_
        init (str, optional): _description_. Defaults to "k-means++".
        random_state (int, optional): _description_. Defaults to 42.
        height (float, optional): _description_. Defaults to 1.5.
        aspect (float, optional): _description_. Defaults to 1.5.

    Returns:
        _type_: _description_

    Example usage:
        n_cluster = 2
        data_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        df = kmeans_with_pairplot(data=df, n_cluster=n_cluster, data_cols=data_cols)
    """
    kmeans = KMeans(n_clusters=n_cluster, init=init, random_state=random_state).fit(
        data
    )
    data["pred"] = kmeans.predict(data)
    sns.pairplot(data, hue="pred", height=height, aspect=aspect, vars=data_cols)
    plt.show()
    return data


def plot_silhouette(X, y_pred):
    """_summary_

    Args:
        X (_type_): _description_
        y_pred (_type_): _description_

    Example usage:
        # Silhouttenkoeffizient nicht annährend 0, hinweis gelungenes clustering
        # außerdem wenn unterschiedliche Länge und Breite, hinweis nicht gelungenes clustering
    """
    cluster_labels = np.unique(y_pred)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_pred, metric="euclidean")
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_pred == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(
            range(y_ax_lower, y_ax_upper),
            c_silhouette_vals,
            height=1.0,
            edgecolor="none",
            color=color,
        )

        yticks.append((y_ax_lower + y_ax_upper) / 2.0)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel("Cluster")
    plt.xlabel("Silhouette coefficient")

    plt.tight_layout()
    # plt.savefig('images/11_04.png', dpi=300)
    plt.show()
