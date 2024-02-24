import tensorflow as tf
import transformers
import torch

Tensor = tf.Tensor


def build_keras_nlp_classifier(
    classifier,
    preset_name: str,
    num_classes: int,
    activation,
    loss,
    optimizer,
    dropout: float = 0.25,
    trainable: bool = False,
    include_prep: bool = False,
):
    """
    Initializes and compiles an NLP classification model based on a specified preset configuration,
    including optional preprocessing layers, custom dropout rate, and training options.

    Args:
        classifier: The classifier class capable of building NLP models. Must have a `from_preset` method.
        preset_name (str): Identifier for the preset configuration to use for model initialization.
        num_classes (int): The number of output classes for the classification task.
        activation: Activation function to use in the output layer of the model.
        loss: Loss function to be used during model training.
        optimizer: Optimization algorithm to be used for training the model.
        dropout (float, optional): Dropout rate to be used in the model. Defaults to 0.25.
        trainable (bool, optional): Flag to set the model's layers as trainable or not. Defaults to False.
        include_prep (bool, optional): Whether to include preprocessing layers in the model. Defaults to False.

    Returns:
        An instance of the model compiled with the specified loss function, optimizer, and accuracy metric.

    Example usage:
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import SparseCategoricalCrossentropy
        from keras_nlp.models import BertClassifier

        classifier = build_nlp_classifier(
            classifier=NLPClassifier,
            preset_name="bert-base-en",
            num_classes=3,
            activation="softmax",
            loss=SparseCategoricalCrossentropy(),
            optimizer=Adam(learning_rate=1e-4),
            dropout=0.5,
            trainable=True,
            include_prep=True
        )

        # Now you can train your classifier on your data
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
    """
    if include_prep:
        classifier = classifier.from_preset(
            preset_name, num_classes=num_classes, activation=activation, dropout=dropout
        )
    else:
        classifier = classifier.from_preset(
            preset_name,
            num_classes=num_classes,
            activation=activation,
            dropout=dropout,
            preprocessor=None,
        )

    classifier.backbone.trainable = trainable

    classifier.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return classifier


def tokenize_features_and_prep_labels(
    texts: list[str],
    labels: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    preset_name: str,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "tf",
) -> tuple[dict, Tensor]:
    """
    Tokenizes input texts and prepares labels for training using using a pre-trained tokenizer from the Hugging Face's Transformers library.
    This function is designed to preprocess data for NLP models, making it ready for model training or evaluation.

    Args:
        texts (list[str]): List of text strings to be tokenized, representing the input data for the NLP model.
        labels (list[int]): List of integer labels associated with each input text, used for model training.
        tokenizer (transformers.PreTrainedTokenizer): An instance of a tokenizer, pre-trained and provided by
            Hugging Face's Transformers library, used to tokenize the input texts.
        preset_name (str): Name of the pre-trained tokenizer to use for tokenizing input texts. This parameter
            is used to ensure that the tokenizer is properly configured if additional setup is required.
        padding (bool, optional): Specifies whether to pad the tokenized input sequences to the longest sequence
            in the batch. Defaults to True.
        truncation (bool, optional): Specifies whether to truncate the tokenized input sequences to the model's
            maximum input length. Defaults to True.
        return_tensors (str, optional): Specifies the type of tensors to return ('tf' for TensorFlow tensors,
            'pt' for PyTorch tensors). Defaults to 'tf'.

    Returns:
        tuple: A tuple containing two elements:
            - encoded_input (dict): A dictionary containing the tokenized input data, ready to be fed into an NLP model.
            - labels (tf.Tensor or torch.Tensor): A tensor of the labels, ready to be used in model training or evaluation.

    Example usage:
        from transformers import BertTokenizer
        texts = ["Hello, world!", "Machine learning is fascinating."]
        labels = [0, 1]
        tokenizer = BertTokenizer

        encoded_input, labels_tensor = tokenize_features_and_prep_labels(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            preset_name='bert-base-uncased',
            padding=True,
            truncation=True,
            return_tensors="tf"
        )

        # Now, `encoded_input` can be used as input to a model, and `labels_tensor` as the target labels for training or evaluation.
    """
    tokenizer = tokenizer.from_pretrained(preset_name)
    encoded_input = tokenizer(
        texts, padding=padding, truncation=truncation, return_tensors=return_tensors
    )
    labels_tensor = (
        tf.constant(labels) if return_tensors == "tf" else torch.tensor(labels)
    )

    return encoded_input, labels_tensor


def build_sequence_classification_model(
    preset_name: str,
    model_task: transformers.PreTrainedModel,
    num_classes: int,
    optimizer,
    loss,
    trainable: bool = True,
) -> transformers.PreTrainedModel:
    """
    Initializes and compiles a sequence classification model using a pre-trained model from the Hugging Face's Transformers library.

    This function allows for the customization of the sequence classification model by specifying the number of classes,
    optimizer, loss function, and whether the model's parameters should be trainable.

    Args:
        preset_name (str): The name of the pre-trained model to be used as the base for the sequence classification task.
        model_task (transformers.PreTrainedModel): The class from Hugging Face's Transformers library representing the model.
        num_classes (int): The number of classes for the classification task.
        optimizer: The optimizer to use for training the model.
        loss: The loss function to use for training the model.
        trainable (bool, optional): Whether the model's weights should be trainable. Defaults to True.

    Returns:
        transformers.PreTrainedModel: The compiled sequence classification model ready for training.

    Example usage:
        from transformers import TFBertForSequenceClassification
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.losses import SparseCategoricalCrossentropy

        model = build_sequence_classification_model(
            preset_name='bert-base-uncased',
            model_task=TFBertForSequenceClassification,
            num_classes=2,
            optimizer=Adam(learning_rate=1e-4),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            trainable=True
        )

        # Model is now compiled and ready for training with your dataset.
    """
    model = model_task.from_pretrained(preset_name, num_labels=num_classes)
    model.trainable = trainable
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
