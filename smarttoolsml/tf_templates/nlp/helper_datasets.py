import tensorflow as tf

def get_tf_datasets_tokenized(x_features, y_labels: list[int], batch_size: int, shuffle: bool = False, seed: int = 42):
    """
    Creates a TensorFlow Dataset for training or evaluation purposes from given features and labels.

    Args:
        x_features (dict): A dictionary containing the tokenized features. Expected keys like 
            'input_ids' and 'attention_mask', which lead to tensors or similar structures accepted by 
            `tf.data.Dataset.from_tensor_slices`.
        y_labels (Tensor, ndarray, list): The labels corresponding to the features. Should be in a format 
            supported by `tf.data.Dataset.from_tensor_slices`.
        batch_size (int): The size of the batches into which the dataset should be divided.
        shuffle (bool, optional): Indicates whether the dataset should be shuffled before batching. 
            Defaults to False.
        seed (int, optional): A seed for the random number generator when `shuffle=True`. Defaults to 42.

    Returns:
        tf.data.Dataset: A TensorFlow `tf.data.Dataset` object ready for training or evaluation. 
        The dataset contains paired features and labels, optionally shuffled and batched.

    Example:
        train_data = get_tf_datasets(X_train, y_train, batch_size=32, shuffle=True)
        val_data = get_tf_datasets(X_val, y_val, batch_size=32)
    """
    data = ({'input_ids': x_features['input_ids'],
             'attention_mask': x_features['attention_mask']}, y_labels)

    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y_labels), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    return dataset


def get_tf_datasets(x_features, y_labels: list[int], batch_size: int, shuffle: bool = False, seed: int = 42):
    """
    Creates a TensorFlow Dataset for training or evaluation purposes from given features and labels.

    Args:
        x_features (dict): 
        y_labels (Tensor, ndarray, list): The labels corresponding to the features. Should be in a format 
            supported by `tf.data.Dataset.from_tensor_slices`.
        batch_size (int): The size of the batches into which the dataset should be divided.
        shuffle (bool, optional): Indicates whether the dataset should be shuffled before batching. 
            Defaults to False.
        seed (int, optional): A seed for the random number generator when `shuffle=True`. Defaults to 42.

    Returns:
        tf.data.Dataset: A TensorFlow `tf.data.Dataset` object ready for training or evaluation. 
        The dataset contains paired features and labels, optionally shuffled and batched.

    Example:
        train_data = get_tf_datasets(X_train, y_train, batch_size=32, shuffle=True)
        val_data = get_tf_datasets(X_val, y_val, batch_size=32)
    """
    data = (x_features, y_labels)

    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y_labels), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    return dataset