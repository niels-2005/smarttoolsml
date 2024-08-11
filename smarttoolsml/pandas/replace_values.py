import pandas as pd 

def replace_values(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_

    Example usage:
        different values in df: yyyMcDonald's, McDonalds
        we want only McDonalds in DataFrame
    """
    df["store_name"] = df["store_name"].replace("yyyMcDonald's", "McDonalds")
    return df 