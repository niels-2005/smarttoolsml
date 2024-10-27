import pandas as pd


def concat_dataframes(list):
    """example list = [df1, df2, df3]"""
    df = pd.concat(list)
    return df
