import matplotlib.pyplot as plt
import pandas as pd


def plot_piechart_with_df_series(
    df_series: pd.DataFrame, title: str = "Pie Chart", figsize: tuple[int, int] = (7, 7)
):
    """
    Plots a pie chart from a Pandas Series.

    Args:
        df_series (pd.Series): The data series to plot, usually counts of categories.
        title (str, optional): Title of the pie chart. Defaults to "Pie Chart".
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (7, 7).

    Example usage:
        plot_piechart_with_df_series(df_series=df["race"])
    """
    counts = df_series.value_counts()

    plt.figure(figsize=figsize)
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title(title)
    plt.show()
