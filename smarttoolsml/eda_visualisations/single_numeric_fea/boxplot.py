import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_boxplot(
    df_series: pd.Series, figsize: tuple[int, int] = (10, 5)
) -> None:
    """_summary_

    Args:
        df_series (pd.Series): _description_
        plot_title (str, optional): _description_. Defaults to "Title".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        plot_single_boxplot(df_series=df["Age"], plot_title="Age Balance", figsize=(10, 5))
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df_series)
    plt.title(f"Boxplot")
    plt.show()


def plot_multi_boxplot(
    df: pd.DataFrame, columns: list, figsize: tuple[int, int] = (10, 5)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        columns = ["Age", "Fare"]
        plot_multi_boxplot(df=df, columns=columns, figsize=(10, 5))
    """
    for col in columns:
        plt.figure(figsize=figsize)
        sns.boxplot(x=col, data=df)
        plt.title(f"Boxplot {col}")
    plt.show()
