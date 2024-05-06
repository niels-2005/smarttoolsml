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
        plot_single_boxplot(df=df, x=x, y=y)
    """
    sns.catplot(x=x, y=y, data=df, kind=kind, height=height, aspect=aspect)
    plt.title(f"Boxplot x={x}, y={y}")
    plt.show()
