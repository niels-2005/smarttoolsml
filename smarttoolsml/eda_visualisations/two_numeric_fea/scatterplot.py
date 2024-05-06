import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_single_scatterplot(
    df: pd.DataFrame, x: str, y: str, figsize: tuple[int, int] = (8, 8)
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (8, 8).

    Example usage:
        x = "Age"
        y = "Fare"
        plot_single_scatterplot(df=df, x=x, y=y)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y, data=df)
    plt.title(f"Scatterplot x={x}, y={y}")
    plt.show()
