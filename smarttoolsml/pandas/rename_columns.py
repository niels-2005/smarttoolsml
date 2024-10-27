import pandas as pd


def rename_columns(df: pd.DataFrame):
    df = df.rename(columns={"A": "a", "B": "b"})
    return df
