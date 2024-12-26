import pandas as pd


def drop_columns_threshold(
    df_train: pd.DataFrame, df_test: pd.DataFrame, threshold: int = 80
):
    """_summary_

    Args:
        df_train (pd.DataFrame): _description_
        df_test (pd.DataFrame): _description_
        threshold (int, optional): _description_. Defaults to 80.

    Returns:
        _type_: _description_

    Example usage:
        threshold = 80

        df_train, df_test = drop_columns_threshold(df_train=df_train, df_test=df_test, threshold=threshold)

        # droppt Columns die zu 80 % aus Null Werten bestehen wegen threshold = 80
    """
    columns = df_train.columns[df_train.isnull().mean() > threshold]

    df_train = df_train.drop(columns=columns)
    df_test = df_test.drop(columns=columns)

    return df_train, df_test


def fill_na_columns(
    df_train: pd.DataFrame, df_test: pd.DataFrame, fill_median: bool = True
):
    """_summary_

    Args:
        df_train (pd.DataFrame): _description_
        df_test (pd.DataFrame): _description_
        fill_median (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df_train, df_test = fill_na_columns(df_train=df_train, df_test=df_test, fill_median=True)

        # Null Werte in categorical Spalten werden mit dem meist vorkommenden Wert aufgefüllt
        # Null Werte in numerischen Spalten werden bei fill_median = True mit dem median aufgefüllt, ansonsten mean
    """
    for column in df_train.columns:
        if df_train[column].isnull().any():
            if df_train[column].dtype == "object":
                mode_value = df_train[column].mode()[0]
                df_train[column] = df_train[column].fillna(mode_value)
                df_test[column] = df_test[column].fillna(mode_value)
            else:
                if fill_median:
                    median_value = df_train[column].median()
                    df_train[column] = df_train[column].fillna(median_value)
                    df_test[column] = df_test[column].fillna(median_value)
                else:
                    mean_value = df_train[column].mean()
                    df_train[column] = df_train[column].fillna(mean_value)
                    df_test[column] = df_test[column].fillna(mean_value)
    return df_train, df_test


def get_nan_rows(df):
    df_nan = df[df.isnull().any(axis=1)]
    return df_nan
