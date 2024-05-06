import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_countplot(
    df: pd.DataFrame, x: str, hue: str, figsize: tuple[int, int] = (8, 4)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        hue (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (8, 4).

    Example usage:
        x = "Sex"
        hue = "Survived"
        plot_single_countplot(df=df, x=x, hue=hue)
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=x, hue=hue, data=df)
    plt.title(f"Countplot for {x}")
    plt.show()


def plot_multi_countplot(
    df: pd.DataFrame, columns: list, hue: str, figsize: tuple[int, int] = (8, 4)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        hue (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (8,4).

    Example usage:
        columns = ["Sex", "Pclass"]
        hue = "Survived"
        plot_multi_countplot(df=df, columns=columns, hue=hue)
    """
    for col in columns:
        plt.figure(figsize=figsize)
        sns.countplot(x=col, hue=hue, data=df)
        plt.title(f"Countplot for {col}")
    plt.show()
