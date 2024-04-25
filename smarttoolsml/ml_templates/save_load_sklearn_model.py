from joblib import dump, load


def save_model(model, filename: str) -> None:
    """
    Saves the specified model to a file using serialization.

    Args:
        model: The model to be saved. This can be any Python object that is serializable.
        filename (str): The path to the file where the model will be saved.

    Example usage:
        save_model(my_model, 'model_filename.pkl')
    """
    dump(model, filename)


def load_model(filename: str):
    """
    Loads a model from a specified file using deserialization.

    Args:
        filename (str): The path to the file from which the model will be loaded.

    Returns:
        The model loaded from the file.

    Example usage:
        loaded_model = load_model('model_filename.pkl')
    """
    model = load(filename)
    return model
