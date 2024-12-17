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


def convert_labels(df: pd.DataFrame, labels_dict: dict, inverse_convert: bool = False):
    """good to use when values are ordinal.

    Args:
        df (pd.DataFrame): _description_
        labels_dict (dict): _description_

    Returns:
        _type_: _description_

    Example usage:
        labels_dict = {"Ham": 0, "Spam": 1} # if column are str
        labels_dict = {0: "Ham", 1: "Spam"} # if column are numeric
        convert_labels(df=df, labels_dict=labels_dict)
    """
    if inverse_convert:
        inv_mapping = {v: k for k, v in labels_dict.items()}
        df["label"].map(inv_mapping)
    else:
        df["label"] = df["label"].map(labels_dict)

    return df


def simple_feature_creation_example(df):
    df["Income per Age"] = df["Income"] / df["Age"]


def convert_ordinal_feature(df):
    # df[size] = [M, L, XL] (...)
    df["x > M"] = df["size"].apply(lambda x: 1 if x in {"L", "XL"} else 0)
    df["x > L"] = df["size"].apply(lambda x: 1 if x == "XL" else 0)
    return df
