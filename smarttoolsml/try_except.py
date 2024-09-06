import pandas as pd


def load_data(url: str) -> pd.DataFrame: 
    try:
        df = pd.read_csv(url, sep=";")
        return df 
    except Exception as e:
        raise e