import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd


def plot_nans_with_df(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (10, 10),
    title: str = "NaN Counts in DataFrame",
    plot_bar: bool = False,
):
    """
    Visualizes missing data in a DataFrame using either a matrix or a bar chart from the missingno library.

    Args:
        df (pd.DataFrame): DataFrame to analyze for missing data.
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): Title of the plot. Defaults to "NaN Counts in DataFrame".
        plot_bar (bool, optional): If True, uses a bar chart to show missing data count per column; otherwise, uses a matrix visualization. Defaults to False.

    Example usage:
        plot_nans_with_df(df=my_dataframe, plot_bar=True)
    """
    plt.figure(figsize=figsize)

    if plot_bar:
        msno.bar(df)
    else:
        msno.matrix(df)

    plt.title(title)
    plt.show()
