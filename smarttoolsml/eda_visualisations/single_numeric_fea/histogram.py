import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_hist(
    df_series: pd.Series, kde: bool = True, height: int = 5, aspect: int = 2
) -> None:
    """_summary_

    Args:
        df_series (pd.Series): _description_
        kde (bool, optional): _description_. Defaults to True.
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        df_series = df["Age"]
        plot_single_hist(df_series=df_series, kde=True)
    """
    sns.displot(df_series, kde=kde, height=height, aspect=aspect)
    plt.title(f"Histogram")
    plt.show()


def plot_multi_hist(
    df: pd.DataFrame, columns: list, kde: bool = True, height: int = 5, aspect: int = 2
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        kde (bool, optional): _description_. Defaults to True.
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        columns = ["Age", "Fare"]
        plot_multi_hist(df=df, columns=columns)
    """
    for col in columns:
        sns.displot(data=df, x=col, kde=kde, height=height, aspect=aspect)
        plt.title(f"Histogram {col}")
    plt.show()
