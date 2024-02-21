import numpy as np
import tensorflow as tf
import tqdm
from tf.keras.preprocessing import image_dataset_from_directory
from tf.keras import Model


def preprocess_dataset(
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
        dataset = dataset.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset


def get_image_datasets(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    preprocess_fn=None,
    include_test_dir: bool = True,
    label_mode: str = "categorical",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    prefetch_buffer=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE,
    drop_remainder=True,
) -> tuple:
    """
    Creates image datasets from directories for training, validation, and optionally testing,
    and applies preprocessing using a specified function.

    Args:
        train_dir (str): Directory with training images.
        valid_dir (str): Directory with validation images.
        test_dir (str): Directory with test images, used only if include_test_dir is True.
        preprocess_fn (callable, optional): The preprocessing function to apply to each element in the dataset.
        include_test_dir (bool, optional): Whether to include test dataset. Defaults to True.
        label_mode (str, optional): Type of label extraction, 'categorical', 'binary', 'int', or None. Defaults to 'categorical'.
        batch_size (int, optional): Size of the batches of data. Defaults to 32.
        image_size (tuple[int, int], optional): The size to resize images to. Defaults to (224, 224).
        prefetch_buffer (int, optional): The prefetch buffer size, typically set to tf.data.AUTOTUNE. Defaults to tf.data.AUTOTUNE.
        num_parallel_calls (int, optional): The number of parallel calls to the map function, typically set to tf.data.AUTOTUNE. Defaults to tf.data.AUTOTUNE.
        drop_remainder (bool, optional): Whether to drop the remainder of the batches if it's not a full batch. Defaults to True.

    Returns:
        tuple: A tuple containing the training dataset, validation dataset, and optionally the test dataset.
    
    Example usage:
        train_dir = '/path/to/train'
        valid_dir = '/path/to/valid'
        test_dir = '/path/to/test'

        train_dataset, valid_dataset, test_dataset = get_image_datasets(
            train_dir=train_dir,
            valid_dir=valid_dir,
            test_dir=test_dir,
            preprocess_fn=preprocess_input, # preprocess_input from resnet
            include_test_dir=True,
            label_mode='categorical',
            batch_size=32,
            image_size=(224, 224)
        )
    """

    # Load datasets from directories
    train_dataset = image_dataset_from_directory(
        train_dir,
        label_mode=label_mode,
        image_size=image_size,
        shuffle=True,
        batch_size=batch_size
    )

    valid_dataset = image_dataset_from_directory(
        valid_dir,
        label_mode=label_mode,
        image_size=image_size,
        shuffle=False, # dont need to shuffle
        batch_size=batch_size,
    )

    # Apply preprocessing, batching, and prefetching
    train_dataset = preprocess_dataset(train_dataset, preprocess_fn, batch_size, prefetch_buffer, num_parallel_calls, drop_remainder)
    valid_dataset = preprocess_dataset(valid_dataset, preprocess_fn, batch_size, prefetch_buffer, num_parallel_calls, drop_remainder)

    # optionally include test_dir
    if include_test_dir:
        test_dataset = image_dataset_from_directory(
            test_dir,
            label_mode=label_mode,
            image_size=image_size,
            shuffle=False, # dont need to shuffle
            batch_size=batch_size, 
        )
        test_dataset = preprocess_dataset(test_dataset, preprocess_fn, batch_size, prefetch_buffer, num_parallel_calls, drop_remainder)
        
        return train_dataset, valid_dataset, test_dataset
    else:
        return train_dataset, valid_dataset


def evaluate_image_classification_model(
    model: Model, files: tf.data.Dataset, is_categorical: bool = True
) -> tuple[float, float, np.ndarray]:
    """
    Evaluates a model on given files and returns predictions, handling both
    categorical and binary classification cases.

    Args:
        model (Model): The trained model to evaluate.
        files (ImageDataset): The dataset or data generator providing the input data.
        is_categorical (bool, optional): Flag to indicate if the model performs
                                         categorical classification. Defaults to True.
                                         If False, the model is assumed to perform binary classification.

    Returns:
        loss (float): The loss value evaluated on the given data.
        accuracy (float): The accuracy value evaluated on the given data.
        y_pred (np.ndarray): The predicted labels for the given data.

    Example usage:
        model_evaluation_img_dataset(model, files, is_categorical=True)
    """
    # Evaluate the model on the provided files
    loss, accuracy = model.evaluate(files, verbose=1)

    # Predict probabilities or labels based on the model type
    y_probs = model.predict(files, verbose=1)

    # Handle categorical and binary cases
    if is_categorical:
        y_pred = np.argmax(y_probs, axis=1)
    else:
        # For binary classification, round the probabilities to get binary labels
        y_pred = np.round(y_probs).astype(int)

    return loss, accuracy, y_pred


def extract_labels_from_dataset(files: tf.data.Dataset) -> np.ndarray:
    """
    Extracts true labels from an ImageDataset.

    This function iterates over an unbatched ImageDataset and extracts the true labels
    for each image. It is assumed that the labels are one-hot encoded and the function
    returns the indices of the maximum values (argmax) as the true labels.

    Args:
        files (ImageDataset): The image dataset from which to extract the true labels.
                              The dataset is expected to tuples of (images, labels).

    Returns:
        np.ndarray: An array of true labels extracted from the dataset.

    Example usage:
        extract_labels_from_dataset(files)
    """
    y_true = []
    for images, labels in tqdm(files.unbatch(), desc="Extracting labels"):
        # Assuming labels are one-hot encoded, get the label indices
        y_true.append(np.argmax(labels.numpy()))

    return np.array(y_true)
