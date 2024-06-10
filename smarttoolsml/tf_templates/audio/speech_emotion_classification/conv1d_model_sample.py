import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_model(X_train):
    model = tf.keras.Sequential(
        [
            L.Conv1D(
                512,
                kernel_size=5,
                strides=1,
                padding="same",
                activation="relu",
                input_shape=(X_train.shape[1], 1),
            ),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Conv1D(512, kernel_size=5, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Dropout(0.2),  # Add dropout layer after the second max pooling layer
            L.Conv1D(256, kernel_size=5, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=5, strides=2, padding="same"),
            L.Dropout(0.2),  # Add dropout layer after the fourth max pooling layer
            L.Conv1D(128, kernel_size=3, strides=1, padding="same", activation="relu"),
            L.BatchNormalization(),
            L.MaxPool1D(pool_size=3, strides=2, padding="same"),
            L.Dropout(0.2),  # Add dropout layer after the fifth max pooling layer
            L.Flatten(),
            L.Dense(512, activation="relu"),
            L.BatchNormalization(),
            L.Dense(7, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    return model


def get_train_data(X: np.ndarray, y: np.ndarray):
    """_summary_

    Args:
        X (np.ndarray): _description_
        y (np.ndarray): _description_

    Returns:
        _type_: _description_

    Example usage:
        X = Emotions.iloc[: ,:-1].values
        y = Emotions['Emotions'].values

        X_train, X_test, y_train, y_test = get_train_data(X, y)
    """
    encoder = OneHotEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test
