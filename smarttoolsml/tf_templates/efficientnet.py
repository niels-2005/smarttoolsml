import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.applications import EfficientNetB0


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
    include_batch_normalisation: bool = False,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """
    Builds a classification model based on a given EfficientNet backbone.

    Args:
        efficientnet (Model): The EfficientNet model to use as the backbone.
        loss: The loss function to use. Can be a string identifier or a TensorFlow loss function.
        optimizer: The optimizer to use. Can be a string identifier or a TensorFlow optimizer instance.
        num_classes (int, optional): The number of classes for the classification task. Required if is_categorical is True.
        is_categorical (bool): If True, uses a softmax activation and expects num_classes to be defined. If False,
                               uses a sigmoid activation, suitable for binary classification.
        include_batch_normalisation (bool): If True, includes a batch normalisation layer after the EfficientNet backbone
                                            and before the final classification layer.
        input_shape (tuple[int, int, int]): The shape of the input images.

    Returns:
        A compiled Keras model ready for training.

    Example usage:
        build_model(EfficientNetB0,
                    loss='categorical_crossentropy',
                    optimizer=Adam(),
                    num_classes=10,
                    is_categorical=True,
                    include_batch_normalisation=False,
                    input_shape=(224, 224, 3))
    """
    # initalize pretrained efficientnet model without top layers and 'avg' pooling
    efficientnet_model = efficientnet(
        include_top=False, input_shape=input_shape, pooling="avg"
    )
    efficientnet_model.trainable = False  # freeze the pretrained layers

    # define model and add layers
    model = Sequential()
    model.add(efficientnet_model)

    # optionally add a batch normalisation layer
    if include_batch_normalisation:
        model.add(layers.BatchNormalization())

    # add Dropout layer
    model.add(layers.Dropout(0.5))

    # Add the final Dense layer for classification
    if is_categorical:
        model.add(layers.Dense(num_classes, activation="softmax"))
    else:
        model.add(layers.Dense(1, activation="sigmoid"))

    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_model_data_augmented(
    efficientnet: Model,
    loss,
    optimizer,
    img_augmentation_layers: list,
    num_classes: int = None,
    is_categorical: bool = False,
    include_batch_normalisation: bool = False,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """
    Builds a classification model based on a given EfficientNet backbone, including image augmentation layers
    and an optional batch normalisation layer.

    Args:
        efficientnet (Model): The EfficientNet model to use as the backbone.
        loss: The loss function to use. Can be a string identifier or a TensorFlow loss function.
        optimizer: The optimizer to use. Can be a string identifier or a TensorFlow optimizer instance.
        img_augmentation_layers (list): List of Keras layers for image augmentation.
        num_classes (int, optional): The number of classes for the classification task. Required if is_categorical is True.
        is_categorical (bool): If True, uses a softmax activation and expects num_classes to be defined. If False,
                               uses a sigmoid activation, suitable for binary classification.
        include_batch_normalisation (bool): If True, includes a batch normalisation layer after the EfficientNet backbone
                                            and before the final classification layer.
        input_shape (tuple[int, int, int]): The shape of the input images.

    Returns:
        A compiled Keras model ready for training.

    Example usage:
        img_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1)
        ]

        model = build_model_data_augmented(EfficientNetB0,
                                           loss='categorical_crossentropy',
                                           optimizer=Adam(),
                                           img_augmentation_layers=img_augmentation_layers,
                                           num_classes=10,
                                           is_categorical=True,
                                           include_batch_normalisation=False,
                                           input_shape=(224, 224, 3))
    """
    model = Sequential()
    model.add(layers.Input(shape=input_shape))  # Define the input shape

    # Add each image augmentation layer to the model
    for layer in img_augmentation_layers:
        model.add(layer)

    # initalize pretrained efficientnet model without top layers and 'avg' pooling
    efficientnet_model = efficientnet(
        include_top=False, pooling="avg", input_shape=input_shape
    )
    efficientnet_model.trainable = False  # Freeze the EfficientNet model
    model.add(efficientnet_model)

    # optionally add a batch normalisation layer
    if include_batch_normalisation:
        model.add(layers.BatchNormalization())

    # add Dropout layer
    model.add(layers.Dropout(0.5))

    # Add the final Dense layer for classification
    if is_categorical:
        model.add(layers.Dense(num_classes, activation="softmax"))
    else:
        model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def unfreeze_model(model: Model, num_layers: int, optimizer, loss) -> Model:
    """
    Unfreezes the top `num_layers` of a efficientnet model for fine-tuning, excluding BatchNormalization layers.

    Args:
        model (Model): The pre-trained efficientnet model to fine-tune.
        num_layers (int): The number of layers from the top to unfreeze.
        optimizer: The optimizer class or instance to use. If a class is provided, `learning_rate` is used to instantiate it.
        loss: The loss function to use for re-compilation of the model.

    Returns:
        The model with unfrozen layers ready for fine-tuning.
    """
    # Unfreeze the specified top layers
    for layer in model.layers[-num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
