import tensorflow as tf

def preprocess_dataset(dataset: tf.data.Dataset, preprocess_fn=None):
    """
    Applies a custom preprocessing function to each element of a given TensorFlow dataset.

    This function is designed to take a TensorFlow dataset object and apply a specified preprocessing function to every element within the dataset.
    The preprocessing function should be capable of handling the dataset's element structure, typically involving operations on image data and labels. 
    It's particularly useful for preparing datasets for model training or inference by applying normalization, resizing, or data augmentation.

    Args:
        dataset (tf.data.Dataset): The dataset to preprocess. It should yield elements that are compatible with the `preprocess_fn`.
        preprocess_fn (callable, optional): A function to apply to each element in the dataset. This function should accept the format of elements the 
        dataset yields, typically (image, label) tuples, and return them in the same format after applying preprocessing.

    Returns:
        tf.data.Dataset: A dataset with the preprocessing function applied to all its elements.

    Example usage:
        def preprocess_fn(image, label):
            return preprocess_input(image), label # preprocess_input from resnet

        train_files = preprocess_dataset(train_files, preprocess_fn=preprocess_fn)
    """
    dataset = dataset.map(lambda image, label: preprocess_fn(image, label))
    return dataset


def preprocess_dataset_optimation(
    dataset: tf.data.Dataset,
    preprocess_fn=None,
    batch_size: int = 32,
    prefetch_buffer=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE,
    drop_remainder=True,
) -> tf.data.Dataset:
    """
    Applies preprocessing, batching, and prefetching to a tf.data.Dataset for performance optimization.

    Args:
        dataset (tf.data.Dataset): The dataset to preprocess.
        preprocess_fn (callable, optional): The preprocessing function to apply to each element in the dataset.
        batch_size (int, optional): Size of the batches of data. Defaults to 32.
        prefetch_buffer (int, optional): The prefetch buffer size, typically set to tf.data.AUTOTUNE for dynamic adjustment.
        num_parallel_calls (int, optional): The number of parallel calls to the map function, typically set to tf.data.AUTOTUNE.
        drop_remainder (bool, optional): Whether to drop the remainder of the batches if it's not a full batch. Defaults to True.

    Returns:
        tf.data.Dataset: The preprocessed dataset.
    """
    if preprocess_fn:
        dataset = dataset.map(
            lambda image, label: preprocess_fn(image, label),
            num_parallel_calls=num_parallel_calls,
        )
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset