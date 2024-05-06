import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_countplot(
    df: pd.DataFrame, x: str, figsize: tuple[int, int] = (8, 5)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (8, 5).

    Example usage:
        x = "Survived"
        plot_single_countplot(df=df, x=x)
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=x, data=df)
    plt.title(f"Countplot of {x}")
    plt.show()


def plot_multi_countplot(
    df: pd.DataFrame,
    columns: list,
    figsize: tuple[int, int] = (8, 5),
    palette: str = "deep",
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (8, 5).
        palette (str, optional): _description_. Defaults to "deep".

    Example usage:
        columns = ["Survived", "Pclass", "Sex"]
        plot_multi_countplot(df=df, columns=columns)
    """
    for col in columns:
        plt.figure(figsize=figsize)
        sns.countplot(x=col, hue=col, data=df, palette=palette)
        plt.title(f"Countplot of {col}")
    plt.show()
