import tensorflow as tf
import transformers


def tokenize_features(
    tokenizer: transformers.PreTrainedTokenizer,
    preset_name: str,
    texts: list[str],
    labels: list[int] = None,
    include_labels: bool = True,
    padding: bool = True,
    truncation: bool = True,
    return_tensors: str = "tf",
) -> tuple:
    """
    Tokenizes input texts and optionally prepares labels for training or evaluation using a pre-trained tokenizer from the Hugging Face's Transformers library. This function is designed to preprocess data for NLP models, making it ready for model training, evaluation, or inference when labels are not provided.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): An instance of a tokenizer, pre-trained and provided by Hugging Face's Transformers library, 
        used to tokenize the input texts.
        preset_name (str): Name of the pre-trained tokenizer to use for tokenizing input texts. This parameter is used to ensure that the tokenizer 
        is properly configured if additional setup is required.
        texts (list[str]): List of text strings to be tokenized, representing the input data for the NLP model.
        labels (list[int], optional): List of integer labels associated with each input text. These are used for model training or evaluation if `include_labels` 
        is True and labels are provided. If no labels are provided, this should be None.
        include_labels (bool, optional): Specifies whether to include labels in the output. If True, labels are processed and returned alongside the tokenized texts. 
        If False, only the tokenized texts are returned. Useful for processing test data without labels. Defaults to True.
        padding (bool, optional): Specifies whether to pad the tokenized input sequences to the longest sequence in the batch. Defaults to True.
        truncation (bool, optional): Specifies whether to truncate the tokenized input sequences to the model's maximum input length. Defaults to True.
        return_tensors (str, optional): Specifies the type of tensors to return ('tf' for TensorFlow tensors). Defaults to 'tf'.

    Returns:
        tuple: A tuple containing one or two elements:
            - encoded_input (dict): A dictionary containing the tokenized input data, ready to be fed into an NLP model.
            - labels (tf.Tensor, optional): A tensor of the labels, ready to be used in model training or evaluation. Only returned if `include_labels` is True and labels are provided.

    Example usage:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer

        X_train = df_train['text'].to_list()
        y_train = df_train['target']

        X_val = df_train['text'].to_list()
        y_val = df_train['target']

        X_test = df_test['text'].to_list()

        # For training data with labels
        X_train, y_train = tokenize_features(
            tokenizer=tokenizer,
            preset_name='bert-base-uncased',
            texts=texts,
            labels=labels,
            include_labels=True
        )

        # For test data without labels
        X_test = tokenize_features(
            tokenizer=tokenizer,
            preset_name='bert-base-uncased',
            texts=texts,
            include_labels=False
        )

        # Now, `encoded_input` can be used as input to a model, and `labels_tensor` (if returned) as the target labels for training or evaluation.
    """
    tokenizer = tokenizer.from_pretrained(preset_name)
    encoded_input = tokenizer(
        texts, padding=padding, truncation=truncation, return_tensors=return_tensors
    )
    if include_labels and labels is not None:
        labels_tensor = tf.constant(labels)
        return encoded_input, labels_tensor
    return encoded_input