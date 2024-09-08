import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns


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


def plot_missing_values_dist(df: pd.DataFrame, threshold: int = 0):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        threshold (int, optional): _description_. Defaults to 0.

    Example usage:
        df = pd.read_csv("...")
        plot_missing_values_dist(df=df, threshold=20)

        threshold = Schwellenwert worüber Columns angezeigt werden
    """
    missing_values = df.isnull().mean() * 100
    missing_values = missing_values[missing_values > threshold].sort_values(
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=missing_values.index, y=missing_values.values, hue=missing_values.index
    )
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Percentage of Missing Values")
    plt.title(f"Missing Values Distribution in df_train (threshold = {threshold})")
    plt.show()
