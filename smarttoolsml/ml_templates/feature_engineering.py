import pandas as pd


def feature_binning(df: pd.DataFrame, col: str, bins: list, labels: list):
    """
    Segment a specified column of a DataFrame into discrete bins and encode the binned data as one-hot encoded dummy variables.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be binned.
        col (str): The column in the DataFrame to apply the binning to.
        bins (list): A list of scalars that defines the bin edges for segmenting the data. Each pair of adjacent values defines an interval.
        labels (list): A list of labels for the bins. The length of this list must be one less than the length of `bins` as each label is assigned to an interval between bin edges.

    Returns:
        pd.DataFrame: A DataFrame with the original data and the new dummy variables appended as additional columns.

    Example usage:
        bins = [0, 6, 20, 55, 80]
        labels = ["very young", "young", "medium", "old"]
        df = feature_binning(df=df, col="Age", bins=bins, labels=labels)
    """
    binning = pd.cut(df[col], bins=bins, labels=labels)
    dummies = pd.get_dummies(binning).astype(int)
    df_dummies = pd.concat([df, dummies], axis=1)
    return df_dummies
