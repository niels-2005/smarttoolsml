import numpy as np
import tensorflow as tf
import transformers


def tokenize_data(
    data: list[str],
    tokenizer: transformers.RobertaTokenizerFast,
    max_len: int = 512,
    add_special_tokens: bool = True,
    padding: str = "max_length",
    return_attention_masks: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenizes a list of text strings using a RoBERTa tokenizer, preparing the data for input into a RoBERTa model.
    This function converts text into sequences of input IDs and generates attention masks to differentiate between real tokens and padding.
    It's specifically tailored for preprocessing data for RoBERTa models.

    Args:
        data (list[str]): A list of text strings to be tokenized. Each string represents a separate input instance.
        tokenizer (transformers.RobertaTokenizerFast): A RoBERTa tokenizer instance pre-configured and ready for use.
        max_len (int, optional): The maximum length of the tokenized sequences.
        add_special_tokens (bool, optional): Whether to add RoBERTa-specific special tokens (like [CLS], [SEP]) to the sequences. Defaults to True.
        padding (str, optional): The padding strategy to apply.
        return_attention_masks (bool, optional): Whether to include attention masks in the output.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two NumPy arrays:
            - The first array contains the input IDs, with each row representing the tokenized text corresponding to each input data string.
            - The second array contains the attention masks for these input IDs, with 1s indicating real tokens and 0s for padding.

    Example usage:
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')
        texts = ["Hello, world!", "RoBERTa is effective."]
        input_ids, attention_masks = tokenize_data(data=texts, tokenizer=tokenizer)

    Note:
        This function is designed explicitly for RoBERTa tokenizers and models.
    """
    input_ids = []
    attention_masks = []
    for text in data:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_len,
            padding=padding,
            return_attention_mask=return_attention_masks,
        )
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
    return np.array(input_ids), np.array(attention_masks)


def create_roberta_model(
    roberta_model: transformers.TFRobertaPreTrainedModel,
    opt: tf.keras.optimizers.Optimizer,
    loss: tf.keras.losses.Loss,
    metric: tf.keras.metrics.Metric,
    num_classes: int,
    is_categorical: bool = True,
    max_len: int = 512,
    trainable_backbone: bool = True,
) -> tf.keras.Model:
    """
    Initializes and compiles a TensorFlow model integrating a pre-trained RoBERTa model with an option to freeze or unfreeze the RoBERTa backbone.
    Adds a dense layer on top for classification and compiles the model with the specified optimizer, loss, and metrics.

    Args:
        roberta_model (transformers.TFRobertaPreTrainedModel): An instance of a pre-trained RoBERTa model.
        opt (tf.keras.optimizers.Optimizer): The optimizer to use during training.
        loss (tf.keras.losses.Loss): The loss function, should be compatible with the type of classification.
        accuracy (tf.keras.metrics.Metric): Metric(s) to use for evaluating performance.
        num_classes (int): Number of classes for the classification task.
        is_categorical (bool, optional): If True, use categorical (softmax) output layer, else binary (sigmoid). Defaults to True.
        max_len (int, optional): Maximum input sequence length. Defaults to 512.
        trainable_backbone (bool, optional): If True, RoBERTa's weights can be updated during training. If False, they are frozen. Defaults to True.

    Returns:
        tf.keras.Model: Compiled Keras model ready for training.

    Example usage:
        roberta_model = TFRobertaModel.from_pretrained("roberta-base")
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        loss = tf.keras.losses.CategoricalCrossentropy()
        accuracy = tf.keras.metrics.CategoricalAccuracy()

        model = create_roberta_model(roberta_model=roberta_base,
                                    opt=opt,
                                    loss=loss,
                                    metric=accuracy,
                                    num_classes=3,
                                    is_categorical=True,
                                    max_len=256,
                                    trainable_backbone=False)

        history = model.fit([train_input_ids, train_attention_masks],
                            y_train,
                            validation_data=([val_input_ids, val_attention_masks], y_val),
                            epochs=5,
                            batch_size=32,
                            callbacks=callbacks)
    """
    # freeze model or not
    roberta_model.trainable = trainable_backbone

    # input layers
    input_ids = tf.keras.Input(shape=(max_len,), dtype="int32", name="input_ids")
    attention_masks = tf.keras.Input(
        shape=(max_len,), dtype="int32", name="attention_masks"
    )

    # roberta model
    output = roberta_model([input_ids, attention_masks])
    output = output[1]

    # categorical or binary classification
    if is_categorical:
        output = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="output"
        )(output)
    else:
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(output)

    # create and compile model
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(optimizer=opt, loss=loss, metrics=metric)
    return model


def tokenize_for_pred(
    text: str,
    tokenizer: transformers.RobertaTokenizerFast,
    max_len: int = 512,
    add_special_tokens: bool = True,
    padding: str = "max_length",
    return_attention_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenizes a single text string for model prediction, converting it into a sequence of input IDs and generating an attention mask.

    Args:
        text (str): The text string to be tokenized. Intended for a single sentence or text snippet.
        tokenizer (transformers.RobertaTokenizerFast): An instance of RoBERTa's tokenizer, pre-configured and ready to use.
        max_len (int, optional): Specifies the maximum length for the tokenized sequence.
        add_special_tokens (bool, optional): If True, special tokens such as [CLS] and [SEP] will be added to the sequence.
        padding (str, optional): The strategy to apply for padding sequences to `max_len`.
        return_attention_mask (bool, optional): If True, generates an attention mask to differentiate real tokens from padding tokens.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays:
            - The first array contains the tokenized input IDs.
            - The second array is the attention mask, with 1s indicating real tokens and 0s indicating padding tokens.

    Example usage:
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')
        text = "Here's a sample text to tokenize."
        input_ids, attention_mask = tokenize_for_pred(text, tokenizer)
    """
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=add_special_tokens,
        max_length=max_len,
        padding=padding,
        return_attention_mask=return_attention_mask,
    )
    return np.array(encoded["input_ids"]), np.array(encoded["attention_mask"])


def predict_on_text(
    model: tf.keras.Model,
    text: str,
    classes: list[str],
    tokenizer: transformers.RobertaTokenizerFast,
    max_len: int = 512,
    add_special_tokens: bool = True,
    padding: str = "max_length",
    return_attention_mask: bool = True,
    is_categorical: bool = True,
) -> None:
    """
    Performs a prediction on a given text using a specified TensorFlow Roberta model and prints the prediction result.
    The function tokenizes the input text according to the provided tokenizer settings,
    feeds the tokenized input to the model, and interprets the model's output as either a class label (for categorical tasks)
    or a binary decision (for binary classification tasks).

    Args:
        model (tf.keras.Model): The pre-trained and compiled TensorFlow model used for making predictions.
        text (str): The text string on which the prediction is to be made. This is the raw text data to be processed.
        classes (list[str]): A list of class names corresponding to the possible outputs of the model.
        tokenizer (transformers.RobertaTokenizerFast): A tokenizer instance compatible with the RoBERTa architecture, used for tokenizing the input text.
        max_len (int, optional): The maximum length to which the input text will be padded or truncated. Defaults to 512.
        add_special_tokens (bool, optional): Specifies whether to add special tokens (e.g., [CLS], [SEP]) to the input, as required by RoBERTa models.
        padding (str, optional): The strategy for padding the input to `max_len`.
        return_attention_mask (bool, optional): Specifies whether to generate an attention mask alongside the input IDs.
        is_categorical (bool, optional): Specifies whether the prediction task is categorical (True) or binary (False).

    Returns:
        None: Instead of returning a value, this function prints the predicted class label.

    Example usage:
        model = load_model('path/to/your/model')
        tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')
        text = "Your example text here."
        classes = ["class_0", "class_1", "class_2"] # For categorical, or ["negative", "positive"] for binary
        predict_on_text(model, text, classes, tokenizer)
    """
    # get tokenized inputs
    input_ids, attention_masks = tokenize_for_pred(
        text=text,
        tokenizer=tokenizer,
        max_len=max_len,
        add_special_tokens=add_special_tokens,
        padding=padding,
        return_attention_mask=return_attention_mask,
    )
    # treat like batches
    pred_prob = model.predict(
        [np.expand_dims(input_ids, axis=0), np.expand_dims(attention_masks, axis=0)]
    )
    # categorical or binary classification
    if is_categorical:
        pred_label = tf.argmax(pred_prob, axis=1).numpy()[0]
        pred_class = classes[pred_label]
    else:
        pred_label = pred_prob.reshape(-1)[0]
        pred_class = classes[int(pred_prob > 0.5)]
    # print the Prediction
    print(f"Pred: {pred_label} ({pred_class})")
    print(f"Text:\n{text}")
