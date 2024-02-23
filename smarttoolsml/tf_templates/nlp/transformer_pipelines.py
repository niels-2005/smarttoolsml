import tensorflow as tf
import transformers


def sequence_classification_pipeline(
    texts: list[str],
    labels: list[int],
    tokenizer: transformers.PreTrainedTokenizer,
    preset_name: str,
    model_task: transformers.PreTrainedModel,
    num_classes: int,
    optimizer,
    loss,
    trainable: bool = True,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "tf",
):
    """
    Prepares an NLP model for training by tokenizing input texts, initializing and
    configuring the model, and then compiling it.

    Args:
        texts (List[str]): List of text strings representing the input data.
        labels (List[int]): List of integer labels associated with the input data.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer class from Hugging Face's Transformers.
        preset_name (str): Name of the pre-trained model to be used.
        model_task (transformers.TFPreTrainedModel): Model class for the specific task.
        num_classes (int): Number of classes for the classification task.
        optimizer (str or tf.keras.optimizers.Optimizer): Optimizer for training the model.
        loss (str or tf.keras.losses.Loss): Loss function for training the model.
        trainable (bool, optional): Specifies whether the model's backbone should be trainable. Defaults to True.
        padding (bool): Whether to pad sequences to the same length.
        truncation (bool): Whether to truncate long sequences.
        return_tensors (str): The format of the returned tensors, typically 'tf'.

    Returns:
        model (TFPreTrainedModel): The prepared and compiled model.
        encoded_input (Dict[tf.Tensor]): Tokenized input data as tensors.
        labels (tf.Tensor): Labels as a tensor.

    Example usage:
        from transformers import BertTokenizerFast, TFBertForSequenceClassification
        from tensorflow.keras.optimizers import Adam

        texts = ["Hello, world!", "Machine learning is amazing."]
        labels = [0, 1]
        tokenizer = BertTokenizerFast
        preset_name = 'bert-base-uncased'
        model_task = TFBertForSequenceClassification

        model, encoded_input, labels = sequence_classification_pipeline(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            preset_name=preset_name,
            model_task=model_task,
            num_classes=2,
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            trainable=True
        )

        model.fit(encoded_input, labels, epochs=2, batch_size=2)
    """
    tokenizer = tokenizer.from_pretrained(preset_name)
    model = model_task.from_pretrained(preset_name, num_labels=num_classes)
    encoded_input = tokenizer(
        texts, padding=padding, truncation=truncation, return_tensors=return_tensors
    )
    model.trainable = trainable
    labels = tf.constant(labels)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model, encoded_input, labels
