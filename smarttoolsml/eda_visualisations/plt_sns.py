import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from pandas.plotting import parallel_coordinates


def plot_corr_heatmap(df: pd.DataFrame, figsize: tuple[int, int] = (15, 15)):
    """
    Plots a correlation heatmap for the numerical columns of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame whose correlations are to be plotted.
        figsize (tuple[int, int], optional): Figure size in inches (width, height). Defaults to (15, 15).

    Example usage:
        plot_corr_heatmap(df=my_dataframe)
    """
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, linewidths=0.5, fmt=".1f", ax=ax)


def plot_line_with_df_series(
    df_series: pd.Series,
    kind: str = "line",
    color: str = "g",
    label: str = "Speed",
    linewidth: int = 1,
    alpha: float = 0.5,
    grid: bool = True,
    linestyle: str = ":",
    figsize: tuple[int, int] = (15, 15),
):
    """
    Plots a line graph from a Pandas Series.

    Args:
        df_series (pd.Series): Series to plot.
        kind (str, optional): Type of plot (e.g., 'line', 'bar'). Defaults to "line".
        color (str, optional): Color of the plot. Defaults to "g" (green).
        label (str, optional): Label for the legend. Defaults to "Speed".
        linewidth (int, optional): Width of the line. Defaults to 1.
        alpha (float, optional): Transparency of the line. Defaults to 0.5.
        grid (bool, optional): Whether to show grid lines. Defaults to True.
        linestyle (str, optional): Style of the line (e.g., '-', '--', ':'). Defaults to ":".
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (15, 15).

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
        figsize=figsize,
    )


def plot_scatter_with_by_df(
    df: pd.DataFrame,
    kind: str = "scatter",
    x: str = "Attack",
    y: str = "Defense",
    alpha: float = 0.5,
    color: str = "red",
    figsize: tuple[int, int] = (15, 15),
):
    """
    Plots a scatter plot for two columns in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        kind (str, optional): Type of plot. Defaults to "scatter".
        x (str, optional): Column name for the x-axis. Defaults to "Attack".
        y (str, optional): Column name for the y-axis. Defaults to "Defense".
        alpha (float, optional): Transparency of the points. Defaults to 0.5.
        color (str, optional): Color of the points. Defaults to "red".
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (15, 15).

    Example usage:
        plot_scatter_with_by_df(df=my_dataframe, x="Attack", y="Defense", color="blue")
    """
    df.plot(kind=kind, x=x, y=y, alpha=alpha, color=color, figsize=figsize)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} {y} {kind} Plot")
    plt.show()


def plot_hist_with_df_series(
    df_series: pd.Series,
    kind: str = "hist",
    bins: int = 50,
    figsize: tuple[int, int] = (10, 6),
    title: str = "Hist",
    cumulative: bool = False,
):
    """
    Plots a histogram from a Pandas Series.

    Args:
        df_series (pd.Series): The data series to plot.
        kind (str, optional): The type of plot to draw. Defaults to "hist".
        bins (int, optional): Number of histogram bins to use. Defaults to 50.
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 6).
        title (str, optional): The title of the plot. Defaults to "Histogram".
        cumulative (bool, optional): If True, plot a cumulative histogram. Defaults to False.

    Example usage:
        plot_hist_with_df_series(df_series=df["Speed"], bins=30)
    """
    df_series.plot(kind=kind, bins=bins, figsize=figsize, cumulative=cumulative)
    plt.title(title)
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


def plot_data_more_df_series(
    df: pd.DataFrame,
    series_list: list = ["Attack", "Defense", "Speed"],
    subplots: bool = True,
    figsize=(5, 5),
):
    """
    Plots multiple data series from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        series_list (list, optional): List of column names to plot. Defaults to ["Attack", "Defense", "Speed"].
        subplots (bool, optional): If True, plot each series in a separate subplot. Defaults to True.
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (5, 5).

    Example usage:
        plot_data_more_df_series(df=df, series_list=["Attack", "Defense", "Speed"], subplots=True)
    """
    data = df.loc[:, series_list]
    data.plot(subplots=subplots, kind="line", figsize=figsize)
    plt.show()


def plot_jointplot(
    df: pd.DataFrame,
    x: str = "area_poverty_ration",
    y: str = "area_highschool_ration",
    kind: str = "kde",
    height: int = 7,
    title: str = "Title",
):
    """
    Creates a joint plot for two variables in a DataFrame, showing their distribution and correlation.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x (str, optional): Column name for the x-axis. Defaults to "area_poverty_ratio".
        y (str, optional): Column name for the y-axis. Defaults to "area_highschool_ratio".
        kind (str, optional): Type of plot to draw, 'scatter', 'reg', 'resid', 'kde', or 'hex'. Defaults to "kde".
        height (int, optional): Size of the figure (both width and height). Defaults to 7.
        title (str, optional): Title of the plot. Defaults to "Joint Density Plot".

    Example usage:
        plot_jointplot(df=df, x="area_poverty_ratio", y="area_highschool_ratio", kind="kde")
    """
    sns.jointplot(x=x, y=y, data=df, kind=kind, height=height)
    plt.title(title)
    plt.show()


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


def plot_lmplot_with_df(
    df: pd.DataFrame,
    x: str = "area",
    y: str = "rate",
    figsize: tuple[int, int] = (10, 10),
    title: str = "Lm Plot",
):
    """
    Plots a linear model plot of two variables from a DataFrame using seaborn's lmplot.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        x (str, optional): The name of the column for the x-axis. Defaults to "area".
        y (str, optional): The name of the column for the y-axis. Defaults to "rate".
        figsize (tuple[int, int], optional): The size of the figure (width, height).
        This setting may not affect lmplot as expected because lmplot manages its own figure. Defaults to (10, 10).
        title (str, optional): The title of the plot. Defaults to "Lm Plot".

    Example usage:
        plot_lmplot_with_df(df=df, x="area", y="rate")
    """
    plt.figure(figsize=figsize)
    sns.lmplot(data=df, x=x, y=y)
    plt.title(title)
    plt.show()


def plot_kdeplot_with_df(
    df: pd.DataFrame,
    x: str = "area",
    y: str = "rate",
    shade: bool = True,
    cut: int = 3,
    figsize: tuple[int, int] = (10, 10),
    title: str = "Kde Plot",
):
    """
    Plots a kernel density estimation (KDE) plot for two variables from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        x (str, optional): The name of the column for the x-axis. Defaults to "area".
        y (str, optional): The name of the column for the y-axis. Defaults to "rate".
        shade (bool, optional): If True, fill the area under the KDE curve. Defaults to True.
        cut (int, optional): Limits the extent of the plot to data within cut standard deviations. Defaults to 3.
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): The title of the plot. Defaults to "Kde Plot".

    Example usage:
        plot_kdeplot_with_df(df=df, x="area", y="rate", shade=True, cut=3)
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(data=df, x=x, y=y, shade=shade, cut=cut)
    plt.title(title)
    plt.show()


def plot_violinplot_with_df(
    df: pd.DataFrame,
    pal,
    inner: str = "points",
    figsize: tuple[int, int] = (10, 10),
    title: str = "Violin Plot",
):
    """
    Plots a violin plot for all numeric columns in a DataFrame, with options for customization.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to plot.
        pal: Color palette to use for the plot. Can be a seaborn palette or a list of colors.
        inner (str, optional): Representation of the datapoints in the violin interior. Defaults to "points".
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): The title of the plot. Defaults to "Violin Plot".

    Example usage:
        pal = sns.cubehelix_palette(2, rot=-0.5, dark=0.3)
        plot_violinplot_with_df(df=df, pal=pal, inner="points")
    """
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, palette=pal, inner=inner)
    plt.title(title)
    plt.show()


def plot_swarmplot_with_df(
    df: pd.DataFrame,
    x: str = "area",
    y: str = "rate",
    figsize: tuple[int, int] = (10, 10),
    title: str = "Swarm Plot",
):
    """
    Plots a swarm plot of two variables from a DataFrame, illustrating the distribution of data points.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x (str, optional): The name of the column for the x-axis. Defaults to "area".
        y (str, optional): The name of the column for the y-axis. Defaults to "rate".
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): The title of the plot. Defaults to "Swarm Plot".

    Example usage:
        plot_swarmplot_with_df(df=df, x="area", y="rate")
    """
    plt.figure(figsize=figsize)
    sns.swarmplot(x=x, y=y, data=df)
    plt.title(title)
    plt.show()


def plot_pairplot_with_df(
    df: pd.DataFrame, figsize: tuple[int, int] = (10, 10), title: str = "Pair Plot"
):
    """
    Plots pair relationships in a DataFrame using seaborn's pairplot function.

    Args:
        df (pd.DataFrame): DataFrame containing the data for plotting.
        figsize (tuple[int, int], optional): The size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): Title for the plots. Defaults to "Pair Plot".

    Example usage:
        plot_pairplot_with_df(df=my_dataframe)
    """
    plt.figure(figsize=figsize)
    sns.pairplot(data=df)
    plt.title(title)
    plt.show()


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


def plot_parallel_coordinates(
    df: pd.DataFrame,
    series: str = "Species",
    figsize: tuple[int, int] = (10, 10),
    title: str = "Parallel Coordinates",
    cmap: str = "Set1",
):
    """
    Plots a parallel coordinates chart for visualizing data across multiple dimensions.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        series (str, optional): Column name to use for differentiating data points by color. Defaults to "Species".
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (10, 10).
        title (str, optional): Title of the plot. Defaults to "Parallel Coordinates".
        cmap (str, optional): Color map to use for different categories in the 'series' column. Defaults to "Set1".

    Example usage:
        plot_parallel_coordinates(df=my_dataframe, series="Category")
    """
    plt.figure(figsize=figsize)
    parallel_coordinates(df, series, colormap=plt.get_cmap(cmap))
    plt.title(title)
    plt.show()


def plot_facetgrid_scatter_with_df(
    df: pd.DataFrame,
    series: str = "Species",
    x: str = "SepalLengthCm",
    y: str = "SepalWidthCm",
    height: int = 4,
    figsize: tuple[int, int] = (10, 10),
):
    """
    Creates a scatter plot grid using seaborn's FacetGrid, colored by categories.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        series (str, optional): Column to categorize data by color. Defaults to "Species".
        x (str, optional): Column name for the x-axis. Defaults to "SepalLengthCm".
        y (str, optional): Column name for the y-axis. Defaults to "SepalWidthCm".
        height (int, optional): Height of each subplot. Defaults to 4.
        figsize (tuple[int, int], optional): Overall size of the figure (width, height). Defaults to (10, 10).

    Example usage:
        plot_facetgrid_scatter_with_df(df=iris_df, series="Species", x="SepalLengthCm", y="SepalWidthCm")
    """
    plt.figure(figsize=figsize)
    sns.FacetGrid(df, hue=series, height=height).map(plt.scatter, x, y).add_legend()
    plt.title()
    plt.show()


def plot_pd_scatter_matrix(
    df: pd.DataFrame,
    color_list: list,
    figsize: tuple[int, int] = (10, 10),
    diagonal: str = "hist",
    alpha: float = 0.5,
    s: int = 200,
    marker: str = "*",
    edgecolors: str = "black",
    title: str = "Scatter Matrix",
):
    """
    Plots a scatter matrix for the DataFrame, providing a pairwise relationships overview among numerical columns.

    Args:
        df (pd.DataFrame): DataFrame to plot.
        color_list (list): List of colors to use for each data point, typically matching the number of categories.
        figsize (tuple[int, int], optional): Size of the figure (width, height). Defaults to (10, 10).
        diagonal (str, optional): Type of plot for the matrix diagonal ('hist' or 'kde'). Defaults to "hist".
        alpha (float, optional): Transparency level of the points. Defaults to 0.5.
        s (int, optional): Size of the markers. Defaults to 200.
        marker (str, optional): Shape of the marker. Defaults to "*".
        edgecolors (str, optional): Color of the marker edges. Defaults to "black".
        title (str, optional): Title of the plot. Defaults to "Scatter Matrix".

    Example usage:
        plot_pd_scatter_matrix(df=my_dataframe, color_list=['red', 'blue', 'green'])
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


def plot_crosstab_by_series(
    df_series_1: pd.Series,
    df_series_2: pd.Series,
    kind: str = "bar",
    figsize: tuple[int, int] = (10, 6),
    title: str = "Title",
):
    pd.crosstab(df_series_1, df_series_2).plot(kind=kind, figsize=figsize)
    plt.title(title)
    plt.xlabel("0 = No Disease, 1 = Disease")
    plt.ylabel("Amount")
    plt.legend(["Female", "Male"])
    plt.show()
