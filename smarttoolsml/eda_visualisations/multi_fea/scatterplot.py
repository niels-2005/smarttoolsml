import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot__scatterplot_3f(
    df: pd.DataFrame, x: str, y: str, hue: str, figsize: tuple[int, int] = (10, 5)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        hue (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        x = "Age"
        y = "Fare"
        hue = "Survived"
        plot__scatterplot_3f(df=df, x=x, y=y, hue=hue)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, hue=hue, data=df)
    plt.show()


def plot_scatterplot_4f(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    size: str,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        hue (str): _description_
        size (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        x = "Age"
        y = "Fare"
        hue = "Survived"
        size = "Sex"
        plot_scatterplot_4f(df=df, x=x, y=y, hue=hue, size=size)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, hue=hue, size=size, data=df)
    plt.show()


def plot_scatterplot_5f(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    size: str,
    style: str,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        hue (str): _description_
        size (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        x = "Age"
        y = "Fare"
        hue = "Survived"
        size = "Sex"
        style = "Pclass"
        plot_scatterplot_5f(df=df, x=x, y=y, hue=hue, size=size, style=style)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, hue=hue, size=size, style=style, data=df)
    plt.show()
