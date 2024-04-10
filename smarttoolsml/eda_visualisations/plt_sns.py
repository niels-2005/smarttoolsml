import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pandas.plotting import parallel_coordinates

def plot_corr_heatmap(df: pd.DataFrame, figsize: tuple[int, int] = (15, 15)):
    corr = df.corr(numeric_only=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, linewidths=0.5, fmt=".1f", ax=ax)


def plot_line_with_df_series(df_series: pd.Series, 
                   kind: str = "line", 
                   color: str = "g", 
                   label: str = "Speed", 
                   linewidth: int = 1, 
                   alpha: float = 0.5, 
                   grid: bool = True, 
                   linestyle: str = ":",
                   figsize: tuple[int, int] = (15, 15)):
    """_summary_

    Args:
        df_series (pd.Series): _description_
        kind (str, optional): _description_. Defaults to "line".
        color (str, optional): _description_. Defaults to "g".
        label (str, optional): _description_. Defaults to "Speed".
        linewidth (int, optional): _description_. Defaults to 1.
        alpha (float, optional): _description_. Defaults to 0.5.
        grid (bool, optional): _description_. Defaults to True.
        linestyle (_type_, optional): _description_. Defaults to ":".

    Example usage:
        plot_df_column(df_series=df["Speed"], kind="line")
        plot_df_column(df_series=df["Defense"], kind="line", color="r", label="Defense")

        plt.legend()
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.title("Line Plot")
        plt.show()
    """
    df_series.plot(
        kind=kind,
        color=color,
        label=label,
        linewidth=linewidth,
        alpha=alpha,
        grid=grid,
        linestyle=linestyle,
        figsize=figsize
    )


def plot_scatter_with_by_df(df: pd.DataFrame, 
               kind: str = "scatter", 
               x: str = "Attack", 
               y: str = "Defense", 
               alpha: float = 0.5, 
               color: str = "red",
               figsize: tuple[int, int] = (15, 15)):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        kind (str, optional): _description_. Defaults to "scatter".
        x (str, optional): _description_. Defaults to "Attack".
        y (str, optional): _description_. Defaults to "Defense".
        alpha (float, optional): _description_. Defaults to 0.5.
        color (str, optional): _description_. Defaults to "red".

    Example usage:
        plot_by_df(df=df, kind="scatter", x="Attack", y="Defense")
    """
    df.plot(kind=kind, x=x, y=y, alpha=alpha, color=color, figsize=figsize)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} {y} {kind} Plot")
    plt.show()


def plot_hist_with_df_series(df_series: pd.Series, 
                             kind: str = "hist", 
                             bins: int = 50, 
                             figsize: tuple[int, int] = (10, 6), 
                             title: str = "Hist",
                             cumulative: bool = False):
    """_summary_

    Args:
        df_series (pd.Series): _description_
        kind (str, optional): _description_. Defaults to "hist".
        bins (int, optional): _description_. Defaults to 50.
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 6).
        title (str, optional): _description_. Defaults to "Hist".

    Example usage:
        plot_hist_with_df_series(df_series=df["Speed"], kind="hist", bins=30)

    """
    df_series.plot(
        kind=kind,
        bins=bins,
        figsize=figsize,
        cumulative=cumulative
    )
    plt.title(title)
    plt.show()


def plot_boxplot_with_df(df: pd.DataFrame, 
                         column: str = "Attack", 
                         by: str = "Legendary", 
                         title: str = "boxplot",
                         figsize: tuple[int, int] = (10, 10)):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        column (str, optional): _description_. Defaults to "Attack".
        by (str, optional): _description_. Defaults to "Legendary".
        title (str, optional): _description_. Defaults to "boxplot".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
    
    Example usage:
        plot_boxplot_with_df(df=df, column="Attack", by="Legendary", figsize=(5, 5))

    """
    df.boxplot(
        column=column,
        by=by,
        figsize=figsize
    )
    plt.title(title)
    plt.show()


def plot_data_more_df_series(df: pd.DataFrame, 
                             series_list: list = ["Attack", "Defense", "Speed"], 
                             subplots: bool = True, 
                             figsize=(5, 5)):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        series_list (list, optional): _description_. Defaults to ["Attack", "Defense", "Speed"].
        subplots (bool, optional): _description_. Defaults to True.
        figsize (tuple, optional): _description_. Defaults to (5, 5).
    
    Example usage:
        plot_data_more_df_series(df=df, series_list=["Attack", "Defense", "Speed"], subplots=True)
    """
    data = df.loc[:, series_list]
    data.plot(subplots=subplots, kind="line", figsize=figsize)
    plt.show()


def plot_jointplot(df: pd.DataFrame, 
                   x: str = "area_poverty_ration", 
                   y: str = "area_highschool_ration", 
                   kind: str = "kde", 
                   height: int = 7,
                   title: str = "Title"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str, optional): _description_. Defaults to "area_poverty_ration".
        y (str, optional): _description_. Defaults to "area_highschool_ration".
        kind (str, optional): _description_. Defaults to "kde".
        height (int, optional): _description_. Defaults to 7.
        title (str, optional): _description_. Defaults to "Title".
    
    Example usage:
        plot_jointplot(df=df, x="area", y="rate", kind="kde")
    """
    
    sns.jointplot(x=x, y=y, data=df, kind=kind, height=height)
    plt.title(title)
    plt.show()


def plot_piechart_with_df_series(df_series: pd.DataFrame, 
                                 title: str = "Pie Chart", 
                                 figsize: tuple[int, int] = (7, 7)):
    """_summary_

    Args:
        df_series (pd.DataFrame): _description_
        title (str, optional): _description_. Defaults to "Pie Chart".
        figsize (tuple[int, int], optional): _description_. Defaults to (7, 7).
    
    Example usage:
        plot_piechart_with_df_series(df_series=kill["race"])
    """
    counts = df_series.value_counts()
    
    plt.figure(figsize=figsize)
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title(title)
    plt.show()


def plot_lmplot_with_df(df: pd.DataFrame, 
                        x: str = "area", 
                        y: str = "rate", 
                        figsize: tuple[int, int] = (10, 10), 
                        title: str = "Lm Plot"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str, optional): _description_. Defaults to "area".
        y (str, optional): _description_. Defaults to "rate".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Lm Plot".

    Example usage:
        plot_lmplot_with_df(df=df, x="area", y="rate")
    """
    plt.figure(figsize=figsize)
    sns.lmplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()


def plot_kdeplot_with_df(df: pd.DataFrame, 
                 x: str = "area", 
                 y: str = "rate", 
                 shade: bool = True, 
                 cut: int = 3, 
                 figsize: tuple[int, int] = (10, 10),
                 title: str = "Kde Plot"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str, optional): _description_. Defaults to "area".
        y (str, optional): _description_. Defaults to "rate".
        shade (bool, optional): _description_. Defaults to True.
        cut (int, optional): _description_. Defaults to 3.
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Kde Plot".
    
    Example usage:
        plot_kdeplot_with_df(df=df, x="area", y="rate", shade=True, cut=3)
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df, x=x, y=y, shade=shade, cut=cut)
    plt.title(title)
    plt.show()


def plot_violinplot_with_df(df: pd.DataFrame, 
                            pal, 
                            inner: str = "points", 
                            figsize: tuple[int, int] = (10, 10), title: str = "Violin Plot"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        pal (_type_): _description_
        inner (str, optional): _description_. Defaults to "points".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Violin Plot".
    
    Example usage:
        pal = sns.cubehelix_palette(2, rot=-0.5, dark=0.3)
        plot_violinplot_with_df(df=df, pal=pal, inner="points")
    """
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, palette=pal, inner=inner)
    plt.title(title)
    plt.show()


def plot_swarmplot_with_df(df: pd.DataFrame,
                           x: str = "area",
                           y: str = "rate",
                           figsize: tuple[int, int] = (10, 10),
                           title: str = "Swarm Plot"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        x (str, optional): _description_. Defaults to "area".
        y (str, optional): _description_. Defaults to "rate".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Swarm Plot".
    
    Example usage:
        plot_swarmplot_with_df(df=df, x="area", y="rate")
    """ 
    plt.figure(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()


def plot_pairplot_with_df(df: pd.DataFrame,
                          figsize: tuple[int, int] = (10, 10),
                          title: str = "Pair Plot"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Pair Plot".
    
    Example usage:
        plot_pairplot_with_df(df=df)
    """
    plt.figure(figsize=figsize)
    sns.pairplot(data=df)
    plt.title(title)
    plt.show()


def plot_nans_with_df(df: pd.DataFrame, 
                            figsize: tuple[int, int] = (10, 10), 
                            title: str = "NaN Counts in DataFrame", 
                            plot_bar: bool = False):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "NaN Count Matrix".
        plot_bar (bool, optional): _description_. Defaults to False.
    
    Example usage:
        plot_nan_matrix_with_df(df=df, plot_bar=False)
    """
    plt.figure(figsize=figsize)

    if plot_bar:
        msno.bar(df)
    else:
        msno.matrix(df)

    plt.title(title)
    plt.show()


def plot_parallel_coordinates(df: pd.DataFrame, 
                              series: str = "Species",
                              figsize: tuple[int, int] = (10, 10), 
                              title: str = "Parallel Coordinates",
                              cmap: str = "Set1"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        series (str, optional): _description_. Defaults to "Species".
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        title (str, optional): _description_. Defaults to "Parallel Coordinates".
        cmap (str, optional): _description_. Defaults to "Set1".
    
    Example usage:
        plot_parallel_coordinates(df=df, series="Species")
    """
    plt.figure(figsize=figsize)
    parallel_coordinates(df, series, colormap=plt.get_cmap(cmap))
    plt.title(title)
    plt.show()


def plot_facetgrid_scatter_with_df(df: pd.DataFrame, 
                                   series: str = "Species", 
                                   x: str = "SepalLengthCm", 
                                   y: str = "SepalWidthCm", 
                                   height: int = 4,
                                   figsize: tuple[int, int] = (10, 10)):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        series (str, optional): _description_. Defaults to "Species".
        x (str, optional): _description_. Defaults to "SepalLengthCm".
        y (str, optional): _description_. Defaults to "SepalWidthCm".
        height (int, optional): _description_. Defaults to 4.
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
    
    Example usage:
        plot_facetgrid_scatter_with_df(df=df, series="Species", x="SepalLengthCm", y="SepalWidthCm")
    """
    plt.figure(figsize=figsize)
    sns.FacetGrid(df, hue=series, height=height).map(
        plt.scatter, x, y
    ).add_legend()
    plt.title()
    plt.show()


def plot_pd_scatter_matrix(df: pd.DataFrame, 
                           color_list: list, 
                           figsize: tuple[int, int] = (10, 10), 
                           diagonal: str = "hist", 
                           alpha: float = 0.5, 
                           s: int = 200, 
                           marker: str = "*", 
                           edgecolors: str = "black",
                           title: str = "Scatter Matrix"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        color_list (list): _description_
        figsize (tuple[int, int], optional): _description_. Defaults to (10, 10).
        diagonal (str, optional): _description_. Defaults to "hist".
        alpha (float, optional): _description_. Defaults to 0.5.
        s (int, optional): _description_. Defaults to 200.
        marker (str, optional): _description_. Defaults to "*".
        edgecolors (str, optional): _description_. Defaults to "black".
    """
    pd.plotting.scatter_matrix(
        df.loc[:, df.columns != "class"],
        c=color_list,
        figsize=figsize,
        diagonal=diagonal,
        alpha=alpha,
        s=s,
        marker=marker,
        edgecolors=edgecolors,
    )
    plt.title(title)
    plt.show()
    
