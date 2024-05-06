import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_kde(
    df_series: pd.Series, kind: str = "kde", height: int = 5, aspect: int = 2
) -> None:
    """_summary_

    Args:
        df_series (pd.Series): _description_
        kde (bool, optional): _description_. Defaults to True.
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        df_series = df["Age"]
        plot_single_kde(df_series=df_series, kde=True)
    """
    sns.displot(df_series, kind=kind, height=height, aspect=aspect)
    plt.show()


def plot_multi_kde(
    df: pd.DataFrame, columns: list, kind: str = "kde", height: int = 5, aspect: int = 2
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        kind (str, optional): _description_. Defaults to "kde".
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        columns = ["Age", "Fare"]
        plot_multi_kde(df=df, columns=columns)
    """
    for col in columns:
        sns.displot(data=df, x=col, kind=kind, height=height, aspect=aspect)
    plt.show()
