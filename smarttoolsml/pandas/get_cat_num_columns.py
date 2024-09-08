import pandas as pd


def get_columns(
    df: pd.DataFrame, print_unique_cat_values: bool = True, print_columns: bool = True
):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        print_unique_cat_values (bool, optional): _description_. Defaults to True.
        print_columns (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv("...")

        cat_cols, num_cols = get_columns(df=df, print_unique_cat_values = True, print_columns = True)
    """
    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include=["number"]).columns

    if print_columns:
        print("Categorical Columns:", cat_cols, "\n")
        print("Numeric Columns:", num_cols, "\n")

    if print_unique_cat_values:
        unique_values = {col: df[col].nunique() for col in cat_cols}
        for col, unique_count in unique_values.items():
            print(f"{col}: {unique_count} unique values")

    return cat_cols, num_cols
