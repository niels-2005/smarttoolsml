import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_corr_heatmap(
    df: pd.DataFrame,
    numeric_only: bool = True,
    annot: bool = True,
    linewidths: float = 0.5,
    fmt: str = ".1f",
    figsize: tuple[int, int] = (15, 15),
):
    """
    Plots a correlation heatmap for the numerical columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame whose correlations are to be plotted.
        figsize (tuple[int, int], optional): Figure size in inches (width, height). Defaults to (15, 15).

    Example usage:
        plot_corr_heatmap(df=my_dataframe)
    """
    corr = df.corr(numeric_only=numeric_only)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, linewidths=linewidths, fmt=fmt, ax=ax)
