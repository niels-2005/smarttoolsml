import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def one_hot_encode_col(
    df: pd.DataFrame, column_name: str, dtype: str = "int"
) -> pd.DataFrame:
    """
    One-hot encodes a specified column in a pandas DataFrame.

    This function applies one-hot encoding to the specified column of the DataFrame using scikit-learn's OneHotEncoder,
    converts the sparse matrix to a dense format, and appends the resulting binary columns back to the original DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be encoded.
        column_name (str): The name of the column in the DataFrame to be one-hot encoded.
        dtype (str, optional): The data type of the encoded columns. Can be 'int', 'float', etc. Defaults to "int".

    Returns:
        pd.DataFrame: A DataFrame with the original data and the new one-hot encoded columns.

    Example usage:
        df = one_hot_encode_col(df=df, column_name="country")
        df.head()
    """
    ohe = OneHotEncoder(dtype=dtype)
    encoded_data_sparse = ohe.fit_transform(df[[column_name]])
    encoded_data = encoded_data_sparse.toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())
    df = pd.concat([df, encoded_df], axis=1)
    return df


def label_encode_col(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Label-encodes a specified column in a pandas DataFrame.

    This function applies label encoding to the specified column of the DataFrame using scikit-learn's LabelEncoder,
    which converts each unique string value into a unique integer. The original column in the DataFrame is replaced
    with its encoded version.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be encoded.
        column_name (str): The name of the column in the DataFrame to be label encoded.

    Returns:
        pd.DataFrame: A DataFrame where the original column is replaced with its encoded version.

    Example:
        df = label_encode_column(df=df, column_name="country")
        df.head()
    """
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df


def label_encode_columns(df: pd.DataFrame, column_names: list[str]):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        column_names (list[str]): _description_

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        features = ["f1", "f2"]
        df = label_encode_columns(df=df, column_names=features)
    """
    df[column_names] = df[column_names].apply(LabelEncoder().fit_transform)
    return df


def simple_get_dummies(df: pd.DataFrame, cat_cols: list[str], drop_first: bool = False):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        cat_cols (list[str]): _description_

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")
        cat_cols = ["f1", "f2"]

        df = simple_get_dummies(df=df, cat_cols=cat_cols)
    """
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df
