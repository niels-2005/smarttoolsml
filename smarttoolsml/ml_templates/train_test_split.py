import pandas as pd
from sklearn.model_selection import train_test_split


def get_X_y(df: pd.DataFrame):
    X = df.drop("output", axis=1)
    y = df["output"]
    return X, y


def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_train_test_df(df):
    df_train, df_test = train_test_split(df, test_size=0.2)
    return df_train, df_test
