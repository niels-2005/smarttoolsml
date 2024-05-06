import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_boxplot_3f(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    kind: str = "box",
    height: int = 5,
    aspect: int = 2,
) -> None:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str): _description_
        y (str): _description_
        hue (str): _description_
        kind (str, optional): _description_. Defaults to "box".
        height (int, optional): _description_. Defaults to 5.
        aspect (int, optional): _description_. Defaults to 2.

    Example usage:
        x = "Survived"
        y = "Age"
        hue = "Pclass"

        # Boxplot
        plot_boxplot_3f(df=df, x=x, y=y, hue=hue)

        # Violinplot
        plot_boxplot_3f(df=df, x=x, y=y, hue=hue, kind="violin")
    """
    sns.catplot(x=x, y=y, hue=hue, kind=kind, data=df, height=height, aspect=aspect)
    plt.show()
