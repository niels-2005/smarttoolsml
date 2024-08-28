import pandas as pd


def print_unique_values(df: pd.DataFrame, only_categorical: bool = True):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        only_categorical (bool, optional): _description_. Defaults to True.
    
    Example usage:
        df = pd.read_csv("...")
        print_unique_values(df=df, only_categorical=True)
    """
    if only_categorical:
        df_columns = df.select_dtypes(include=['object']).columns 
    else:
        df_columns = df.columns

    unique_values = {col: df[col].nunique() for col in df_columns}
    
    for col, unique_count in unique_values.items():
        print(f"{col}: {unique_count} unique values")