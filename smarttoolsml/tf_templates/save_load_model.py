import pickle

from tensorflow.keras.models import model_from_json


def save_model_json_weights(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_weights.h5")


def load_model_json_weigths():
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model_weights.h5")
    loaded_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return loaded_model


# Additional Functions


def save_scaler(scaler):
    """_summary_

    Args:
        scaler (_type_): _description_

    Example usage:
        scaler = StandardScaler()
        save_scaler(scaler)
    """
    with open("scaler.pickle", "wb") as f:
        pickle.dump(scaler, f)


def load_scaler():
    with open("scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
        return scaler


def save_encoder(encoder):
    """_summary_

    Args:
        encoder (_type_): _description_

    Example usage:
        encoder = OneHotEncoder()
        save_encoder(encoder)
    """
    with open("encoder.pickle", "wb") as f:
        pickle.dump(encoder, f)


def load_encoder():
    with open("encoder.pickle", "rb") as f:
        encoder = pickle.load(f)
        return encoder
