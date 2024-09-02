from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def simple_impute_df(df: pd.DataFrame, strategy: str = "mean"):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        strategy (str, optional): _description_. Defaults to "mean".

    Returns:
        _type_: _description_
    
    Example usage:
        df = pd.read_csv("...")
        df_imp = simple_impute_df(df=df, strategy = "mean")
    """
    si = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_imp = pd.DataFrame(si.fit_transform(df), columns=df.columns)
    return df_imp


def simple_impute_column(df: pd.DataFrame, df_column: str, strategy: str = "mean", add_indicator: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        df_column (str): _description_
        strategy (str, optional): _description_. Defaults to "mean".
        add_indicator (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        df_column = "age"

        df = simple_imput_column(df=df, df_column=df_column, strategy="mean", add_indicator=True)

        # add_indicator = True fügt eine zusätzliche Spalte hinzu auf Basis ob Wert vorhanden war oder nicht
    """
    si = SimpleImputer(missing_values=np.nan, strategy=strategy, add_indicator=add_indicator)

    if add_indicator:
        indicator_column = df_column + "_indicator"
        df[[df_column, indicator_column]] = si.fit_transform(df[[df_column]])
    else:
        df[[df_column]] = si.fit_transform(df[[df_column]])

    return df


def knn_impute_df(df: pd.DataFrame, weights: str = "distance", n_neighbors: int = 5):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        strategy (str, optional): _description_. Defaults to "mean".

    Returns:
        _type_: _description_
    
    Example usage:
        df = pd.read_csv("...")
        df_imp = knn_impute_df(df=df,weights="distance", n_neighbors=10)
    """
    ki = KNNImputer(missing_values=np.nan, weights=weights, n_neighbors=n_neighbors)
    df_imp = pd.DataFrame(ki.fit_transform(df), columns=df.columns)
    return df_imp


def knn_impute_column(df: pd.DataFrame, df_column: str, n_neighbors: int = 5, weights: str = "distance", add_indicator: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        df_column (str): _description_
        weights (str, optional): _description_. Defaults to "distance".
        add_indicator (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    
    Example usage:
        df = pd.read_csv("...")
        df_column = "age"

        df = knn_impute_column(df=df, df_column=df_column, strategy="mean", add_indicator=True)

        # add_indicator = True fügt eine zusätzliche Spalte hinzu auf Basis ob Wert vorhanden war oder nicht
    """
    ki = KNNImputer(missing_values=np.nan, weights=weights, add_indicator=add_indicator, n_neighbors=n_neighbors)

    if add_indicator:
        indicator_column = df_column + "_indicator"
        df[[df_column, indicator_column]] = ki.fit_transform(df[[df_column]])
    else:
        df[[df_column]] = ki.fit_transform(df[[df_column]])

    return df


def iterative_impute_df(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    
    Example usage:
        df = pd.read_csv("...")
        df_imp = iterative_impute_df(df=df)
    """
    estimator = RandomForestRegressor()
    iirf = IterativeImputer(missing_values = np.nan, estimator=estimator)
    df_imp = pd.DataFrame(iirf.fit_transform(df), columns=df.columns)
    return df_imp


def iterative_impute_column(df: pd.DataFrame, df_column: str, add_indicator: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        df_column (str): _description_

    Returns:
        _type_: _description_
    """
    estimator = RandomForestRegressor()
    iirf = IterativeImputer(missing_values = np.nan, estimator=estimator)

    if add_indicator:
        indicator_column = df_column + "_indicator"
        df[[df_column, indicator_column]] = iirf.fit_transform(df[[df_column]])
    else:
        df[[df_column]] = iirf.fit_transform(df[[df_column]])
    
    return df