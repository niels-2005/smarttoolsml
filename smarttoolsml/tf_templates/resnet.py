import tensorflow as tf 
from tf.keras import Model, Sequential, layers


def build_model(
    resnet: Model,
    loss,
    optimizer,
    num_classes: int = None,
    is_categorical: bool = True,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """
    Builds a classification model based on a given ResNet backbone.

    This function initializes a ResNet model without the top layers, adds a Dense layer for feature extraction,
    and includes a final Dense layer for classification. The ResNet model's layers are frozen to retain
    the pretrained weights. The function then compiles the model with the specified loss function and optimizer.

    Args:
        resnet (Model): The ResNet model to use as the backbone. This should be a callable that returns a
                        TensorFlow/Keras model instance.
        loss: The loss function to use for the model. Can be a string identifier or a TensorFlow loss function.
        optimizer: The optimizer to use for the model. Can be a string identifier or a TensorFlow optimizer instance.
        num_classes (int, optional): The number of classes for the classification task. Required if is_categorical is True.
        is_categorical (bool): Determines the type of classification. If True, the function adds a softmax
                               activation layer suitable for multi-class classification. If False, it adds a
                               sigmoid activation layer for binary classification.
        input_shape (tuple[int, int, int]): The shape of the input data that the model should expect. The default
                                            is set to (224, 224, 3), which is the standard input size for ResNet models.

    Returns:
        Model: A compiled TensorFlow/Keras model ready for training.

    Example usage:
        build_model(ResNet50,
                    loss='categorical_crossentropy',
                    optimizer='adam',
                    num_classes=10,
                    is_categorical=True,
                    input_shape=(224, 224, 3))
    """
    # initalize pretrained resnet model without top layers and 'avg' pooling
    resnet_model = resnet(
        include_top=False, input_shape=input_shape, pooling="avg"
    )
    resnet_model.trainable = False  # freeze the pretrained layers

    # define model and add layers
    model = Sequential()
    model.add(resnet_model)
    model.add(layers.Dense(1024, activation='relu'))

    # Add the final Dense layer for classification
    if is_categorical:
        model.add(layers.Dense(num_classes, activation='softmax'))
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
    
    # compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def build_model_data_augmented(
    resnet: Model,
    loss,
    optimizer,
    img_augmentation_layers: list,
    num_classes: int = None,
    is_categorical: bool = False,
    input_shape: tuple[int, int, int] = (224, 224, 3),
) -> Model:
    """
    Builds a classification model based on a given ResNet backbone, including image augmentation layers.

    This function initializes a ResNet model without the top layers and integrates image augmentation layers
    directly into the model. The ResNet model's layers are frozen to retain the pretrained weights. It then adds
    a Dense layer for feature extraction and includes a final Dense layer for classification. The function compiles
    the model with the specified loss function and optimizer, making it ready for training.

    Args:
        resnet (Model): The ResNet model to use as the backbone. This should be a callable that returns a
                        TensorFlow/Keras model instance.
        loss: The loss function to use for the model. Can be a string identifier or a TensorFlow loss function.
        optimizer: The optimizer to use for the model. Can be a string identifier or a TensorFlow optimizer instance.
        img_augmentation_layers (list): List of Keras layers for image augmentation to be applied before feeding
                                        images into the ResNet backbone.
        num_classes (int, optional): The number of classes for the classification task. Required if is_categorical is True.
        is_categorical (bool): Determines the type of classification. If True, uses a softmax activation layer
                               suitable for multi-class classification. If False, uses a sigmoid activation layer
                               for binary classification.
        input_shape (tuple[int, int, int]): The shape of the input data that the model should expect. The default
                                            is set to (224, 224, 3), which is the standard input size for ResNet models.

    Returns:
        Model: A compiled TensorFlow/Keras model ready for training.

    Example usage:
        img_augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
            layers.RandomContrast(factor=0.1),
            layers.RandomBrightness(factor=0.1)
        ]

        model = build_model_data_augmented(ResNet50,
                                           loss='categorical_crossentropy',
                                           optimizer=Adam(),
                                           img_augmentation_layers=img_augmentation_layers,
                                           num_classes=10,
                                           is_categorical=True,
                                           input_shape=(224, 224, 3))
    """
    model = Sequential()
    model.add(layers.Input(shape=input_shape))  # Define the input shape

    # Add each image augmentation layer to the model
    for layer in img_augmentation_layers:
        model.add(layer)

    # initalize pretrained resnet model without top layers and 'avg' pooling
    resnet_model = resnet(
        include_top=False, pooling="avg", input_shape=input_shape
    )
    resnet_model.trainable = False  # Freeze the ResNet model
    model.add(resnet_model)
    model.add(layers.Dense(1024, activation='relu'))

    # Add the final Dense layer for classification
    if is_categorical:
        model.add(layers.Dense(num_classes, activation="softmax"))
    else:
        model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    return model


def unfreeze_model(model: Model, num_layers: int, optimizer, loss, freeze_batch_norm_layer: bool = False) -> Model:
    """
    Unfreezes the top `num_layers` of a model for fine-tuning. Can optionally keep BatchNormalization layers frozen.

    Args:
        model (Model): The pre-trained model to fine-tune.
        num_layers (int): The number of layers from the top to unfreeze.
        optimizer: The optimizer to use. Can be a class or an instance. If a class is provided, it should
                   be instantiated with a learning rate before being passed to this function.
        loss: The loss function to use for the re-compilation of the model.
        freeze_batch_norm_layer (bool): If True, keeps BatchNormalization layers frozen while unfreezing the specified
                                        top layers. If False, all specified top layers are unfrozen, including BatchNormalization layers.

    Returns:
        Model: The model with unfrozen layers ready for fine-tuning.
    """
    # Unfreeze the specified top layers
    for layer in model.layers[-num_layers:]:
        if freeze_batch_norm_layer and isinstance(layer, layers.BatchNormalization):
            continue  # Skip unfreezing BatchNormalization layers if freeze_batch_norm_layer is True
        layer.trainable = True

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
