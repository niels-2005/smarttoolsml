import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplot(
    df_series: pd.Series, plot_title: str = "Title", figsize: tuple[int, int] = (10, 5)
) -> None:
    """_summary_

    Args:
        df_series (pd.Series): _description_
        plot_title (str, optional): _description_. Defaults to "Title".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 5).

    Example usage:
        plot_boxplot(df_series=df["Age"], plot_title="Age Balance", figsize=(10, 5))
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=df_series)
    plt.title(plot_title)
    plt.show()


def plot_boxplot_with_df(
    df: pd.DataFrame,
    column: str = "Attack",
    by: str = "Legendary",
    title: str = "boxplot",
    figsize: tuple[int, int] = (10, 10),
):
    """
    Creates a boxplot for the specified column grouped by another column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str, optional): The column to create a boxplot for. Defaults to "Attack".
        by (str, optional): The column to group data by for the boxplots. Defaults to "Legendary".
        title (str, optional): The title of the plot. Defaults to "Boxplot".
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 10).

    Example usage:
        plot_boxplot_with_df(df=df, column="Attack", by="Legendary")
    """
    df.boxplot(column=column, by=by, figsize=figsize)
    plt.title(title)
    plt.show()
