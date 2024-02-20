import tensorflow as tf
from tf.keras import Model, Sequential, layers
from tf.keras.applications import EfficientNetB0


def print_efficientnet_resolutions():
    print("EfficientNetB0: (224, 224)")
    print("EfficientNetB1: (240, 240)")
    print("EfficientNetB2: (260, 260)")
    print("EfficientNetB3: (300, 300)")
    print("EfficientNetB4: (380, 380)")
    print("EfficientNetB5: (456, 456)")
    print("EfficientNetB6: (528, 528)")
    print("EfficientNetB7: (600, 600)")
    print("https://keras.io/api/applications/")


def build_model(
    efficientnet: Model,
    loss,
    optimizer,
    num_classes: int = None,
    is_categorical: bool = True,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """
    Builds a classification model based on a given EfficientNet backbone.

    Args:
        efficientnet (Model): The EfficientNet model to use as the backbone.
        input_shape (tuple[int, int, int]): The shape of the input images.
        num_classes (int): The number of classes for the categorical classification task.
        loss: The loss function to use. Can be a string identifier or a TensorFlow loss function.
        optimizer: The optimizer to use. Can be a string identifier or a TensorFlow optimizer instance.

    Returns:
        A compiled Keras model ready for training.

    Example usage:
        build_model(EfficientNetB0,
                    loss='categorical_crossentropy',
                    optimizer='Adam()',
                    num_classes=10)
    """

    if is_categorical and num_classes is None:
        raise ValueError(
            "num_classes must be specified for categorical classification."
        )

    efficientnet_model = efficientnet(
        include_top=False, input_shape=input_shape, pooling="avg"
    )
    efficientnet_model.trainable = False

    model = Sequential()
    model.add(efficientnet_model)
    model.add(layers.Dropout(0.5))

    if is_categorical:
        model.add(layers.Dense(num_classes, activation="softmax"))
    else:
        model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model
