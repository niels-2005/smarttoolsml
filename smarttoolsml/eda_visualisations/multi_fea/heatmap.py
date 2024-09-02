import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dython.nominal import associations

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
    plt.title('Correlation Matrix without Categorical Features')
    plt.show()


def plot_corr_heatmap_with_cat_features(    
    df: pd.DataFrame,
    annot: bool = True,
    linewidths: float = 0.5,
    fmt: str = ".1f",
    figsize: tuple[int, int] = (15, 15),):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        annot (bool, optional): _description_. Defaults to True.
        linewidths (float, optional): _description_. Defaults to 0.5.
        fmt (str, optional): _description_. Defaults to ".1f".
        figsize (tuple[int, int], optional): _description_. Defaults to (15, 15).
    """
    asso_df = associations(df, nominal_columns="all", plot=False)
    corr_matrix = asso_df["corr"]
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt=fmt, linewidths=linewidths)
    plt.title('Correlation Matrix including Categorical Features')
    plt.show()


