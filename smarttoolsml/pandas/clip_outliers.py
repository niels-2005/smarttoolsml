import pandas as pd

def clip_outliers(df: pd.DataFrame, numerical_cols: list, quantile_range: list = [0.05, 0.95]):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        numerical_cols (list): _description_

    Returns:
        _type_: _description_
    
    Example usage:
        df = pd.read_csv("...")
        numerical_cols = ["f1", "f2"]
        quantile_range = [0.05, 0.95]

        df = clip_outliers(df=df, numerical_cols=numerical_cols, quantile_range=quantile_range)

        Was passiert hier?

        Alle Werte die unter dem 5% oder über den 95% Quantil liegen, bekommen den gleiche Wert wie 
        der 5% oder 95% Quantil
    """
    df[numerical_cols] = df[numerical_cols].apply(lambda x: x.clip(*x.quantile(quantile_range)))
    return df
