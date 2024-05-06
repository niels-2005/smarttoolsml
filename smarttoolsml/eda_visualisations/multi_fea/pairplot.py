import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pairplot(
    df: pd.DataFrame,
    hue: str,
    palette: str = "seismic",
    diag_kind: str = "kde",
    height: float = 1.5,
    aspect: float = 1.5,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        hue (str): _description_
        palette (str, optional): _description_. Defaults to "seismic".
        diag_kind (str, optional): _description_. Defaults to "kde".
        height (float, optional): _description_. Defaults to 1.5.
        aspect (float, optional): _description_. Defaults to 1.5.

    Example usage:
        hue = "Survived"
        plot_pairplot(df=df, hue=hue)
    """
    sns.pairplot(
        data=df,
        hue=hue,
        palette=palette,
        height=height,
        diag_kind=diag_kind,
        aspect=aspect,
    )
    plt.show()


def plot_pairplot_vars(
    df: pd.DataFrame,
    hue: str,
    data_cols: list,
    height: float = 1.5,
    aspect: float = 1.5,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        hue (str): _description_
        data_cols (list): _description_
        height (float, optional): _description_. Defaults to 1.5.
        aspect (float, optional): _description_. Defaults to 1.5.

    Example usage:
        data_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        hue = "species"
        plot_pairplot_vars(df=iris, hue=hue, data_cols=data_cols)
    """
    sns.pairplot(data=df, hue=hue, height=height, aspect=aspect, vars=data_cols)
    plt.show()
