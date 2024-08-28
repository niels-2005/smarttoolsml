import numpy as np
import pandas as pd


def convert_string_to_datetime(
    df: pd.DataFrame, col: str, format: str = "%d-%m-%Y"
) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df[col], format=format)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df


def convert_string_to_ordinal(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df["date_ordinal"] = df[col].apply(lambda x: x.toordinal())
    return df
