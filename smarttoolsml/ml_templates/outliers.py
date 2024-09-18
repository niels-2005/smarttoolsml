from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation


def cut_numerical_columns(
    df: pd.DataFrame, quantile_range: list[int, int], drop_duplicates: bool = True
):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        quantile_range (list[int, int]): _description_
        drop_duplicates (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        quantile_range = [0.05, 0.95]

        df = cut_numerical_columns(df=df, quantile_range=quantile_range, drop_duplicates=True)
    """
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].apply(lambda x: x.clip(*x.quantile(quantile_range)))
    if drop_duplicates:
        df = df.drop_duplicates()
    return df


def iqr_method(df: pd.DataFrame, n: int, features: list, drop_outliers: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n (int): _description_
        features (list): _description_
        drop_outliers (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        n = 1
        features = ["f1", "f2", "f3"]

        df = iqr_method(df=df, n=n, features=features, drop_outliers=True)
    """
    outlier_list = []

    for column in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR
        # Determining a list of indices of outliers
        outlier_list_column = df[
            (df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step)
        ].index
        # appending the list of outliers
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] < Q1 - outlier_step]
    df2 = df[df[column] > Q3 + outlier_step]

    print("Total number of outliers is:", df1.shape[0] + df2.shape[0])

    if drop_outliers:
        df = df.drop(multiple_outliers).reset_index(drop=True)
        return df


def stdev_method(df: pd.DataFrame, n: int, features: list, drop_outliers: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n (int): _description_
        features (list): _description_
        drop_outliers (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        n = 1
        features = ["f1", "f2", "f3"]

        df = stdev_method(df=df, n=n, features=features, drop_outliers=True)
    """
    outlier_indices = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        # calculate the cutoff value
        cut_off = data_std * 3
        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[
            (df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)
        ].index
        # appending the found outlier indices for column to the list of outlier indices
        outlier_indices.extend(outlier_list_column)
    # selecting observations containing more than x outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > data_mean + cut_off]
    df2 = df[df[column] < data_mean - cut_off]
    print("Total number of outliers is:", df1.shape[0] + df2.shape[0])

    if drop_outliers:
        df = df.drop(multiple_outliers).reset_index(drop=True)
        return df


def z_score_method(
    df: pd.DataFrame, n: int, features: list, drop_outliers: bool = True
):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n (int): _description_
        features (list): _description_
        drop_outliers (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        n = 1
        features = ["f1", "f2", "f3"]

        df = z_score_method(df=df, n=n, features=features, drop_outliers=True)
    """
    outlier_list = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        threshold = 3
        z_score = abs((df[column] - data_mean) / data_std)
        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[z_score > threshold].index
        # appending the found outlier indices for column to the list of outlier indices
        outlier_list.extend(outlier_list_column)
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)
    # Calculate the number of outlier records
    df1 = df[z_score > threshold]
    print("Total number of outliers is:", df1.shape[0])

    if drop_outliers:
        df = df.drop(multiple_outliers).reset_index(drop=True)
        return df


def z_score_method_modified(
    df: pd.DataFrame, n: int, features: list, drop_outliers: bool = True
):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n (int): _description_
        features (list): _description_
        drop_outliers (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        n = 1
        features = ["f1", "f2", "f3"]

        df = z_score_method_modified(df=df, n=n, features=features, drop_outliers=True)
    """
    outlier_list = []

    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        threshold = 3
        MAD = median_abs_deviation

        mod_z_score = abs(0.6745 * (df[column] - data_mean) / MAD(df[column]))

        # Determining a list of indices of outliers for feature column
        outlier_list_column = df[mod_z_score > threshold].index

        # appending the found outlier indices for column to the list of outlier indices
        outlier_list.extend(outlier_list_column)

    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)
    multiple_outliers = list(k for k, v in outlier_list.items() if v > n)

    # Calculate the number of outlier records
    df1 = df[mod_z_score > threshold]
    print("Total number of outliers is:", df1.shape[0])

    if drop_outliers:
        df = df.drop(multiple_outliers).reset_index(drop=True)
        return df
