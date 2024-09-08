import pandas as pd


def group_by(df: pd.DataFrame):
    """_summary_

    grouped by Location Temperature

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    grouped = df.groupby("Location")["Temperature"].agg(["mean", "min", "max", "count"])
    return grouped
