import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    kind: str = "box",
    height: int = 5,
    aspect: int = 2,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        kind (str, optional): _description_. Defaults to "box".
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        x = "Sex"
        y = "Age"

        # Boxplot
        plot_single_boxplot(df=df, x=x, y=y)

        # Violinplot
        plot_single_boxplot(df=df, x=x, y=y, kind="violin")
    """
    sns.catplot(x=x, y=y, data=df, kind=kind, height=height, aspect=aspect)
    plt.title(f"Boxplot x={x}, y={y}")
    plt.show()


def plot_multi_boxplot(
    df: pd.DataFrame,
    x_cols: list,
    y: str,
    kind: str = "box",
    height: int = 5,
    aspect: int = 2,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x_cols (list): _description_
        y (str): _description_
        kind (str, optional): _description_. Defaults to "box".
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        x_cols = ["Sex", "Pclass"]
        y = "Age"

        # Boxplot
        plot_single_boxplot(df=df, x_cols=x_cols, y=y)

        # Violinplot
        plot_single_boxplot(df=df, x_cols, y=y, kind="violin")
    """
    for col in x_cols:
        sns.catplot(x=col, y=y, data=df, kind=kind, height=height, aspect=aspect)
        plt.title(f"Boxplot x={col}, y={y}")
    plt.show()
