import pandas as pd 

def get_predictions_as_df(model, X_test, y_true, encoder):
    """_summary_

    Args:
        model (_type_): _description_
        X_test (_type_): _description_
        y_true (_type_): _description_
        encoder (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        encoder = OneHotEncoder()
        model = Model()
        
        df = get_predictions_as_df(model, X_test, y_true, encoder=encoder)
    """
    y_pred = model.predict(X_test)
    y_pred = encoder.inverse_transform(y_pred)
    y_true = encoder.inverse_transform(y_true)

    df = pd.DataFrame(columns=["Predicted Labels", "True Labels"])
    df["Predicted Labels"] = y_pred.flatten()
    df["True Labels"] = y_true.flatten()

    return df

