import pandas as pd


def drop_specific_rows(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        indices (list[int]): _description_

    Returns:
        pd.DataFrame: _description_

    Example usage:
        indices = df[df["gender] == "Other"].index

        or

        indices = df[((df.Name == 'jhon') &( df.Age == 15) & (df.Grade == 'A'))].index

        df = drop_specific_rows(df=df, indices=indices)
    """
    df = df.drop(indices)
    return df
