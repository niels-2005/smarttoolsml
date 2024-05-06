import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


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
