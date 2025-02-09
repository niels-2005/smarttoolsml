import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def scale_training_data(X_train, X_test, scaler):
    """_summary_

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        scaler (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        X_train = [...]
        X_test = [...]
        scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler = RobustScaler()

        X_train, X_test = scale_training_data(X_train=X_train, X_test=X_test, scaler=scaler)
    """
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def scale_columns(df, columns: list, scaler):
    """
    Example usage:
        scaler = StandardScaler()
        scaler = MinMaxScaler()
        scaler = RobustScaler()

        columns = ["Age", "Income"]

        df = scale_columns(df, columns)
    """
    df[columns] = scaler.fit_transform(df[columns])
    return df


# standardize: (arr - arr.mean()) / arr.std()
# normalize: (arr - arr.min()) / (arr.max() - arr.min())


def scale_mone_tone(X):
    """_summary_

    Args:
        X (_type_): _description_

    Example usage:
        X = [...]
        X = scale_mone_tone(X)

        scale X to [-1, 1]
    """
    X = ((X / 255.0) - 0.5) * 2
    return X
