from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def scale_data(X_train, X_test, scaler):
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

        X_train, X_test = scale_data(X_train=X_train, X_test=X_test, scaler=scaler)
    """
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test
