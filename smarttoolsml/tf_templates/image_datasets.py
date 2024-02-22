import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Model


def get_image_datasets(
    train_dir: str,
    valid_dir: str,
    test_dir: str,
    include_test_dir: bool = True,
    label_mode: str = "categorical",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> tuple:
    """
    Creates image datasets from directories for training, validation, and optionally testing

    Args:
        train_dir (str): Directory with training images.
        valid_dir (str): Directory with validation images.
        test_dir (str): Directory with test images, used only if include_test_dir is True.
        include_test_dir (bool, optional): Whether to include test dataset. Defaults to True.
        label_mode (str, optional): Type of label extraction, 'categorical', 'binary', 'int', or None. Defaults to 'categorical'.
        batch_size (int, optional): Size of the batches of data. Defaults to 32.
        image_size (tuple[int, int], optional): The size to resize images to. Defaults to (224, 224).

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
        batch_size=batch_size,
    )

    class_names = train_dataset.class_names

    valid_dataset = image_dataset_from_directory(
        valid_dir,
        label_mode=label_mode,
        image_size=image_size,
        shuffle=False,  # dont need to shuffle
        batch_size=batch_size,
    )

    # optionally include test_dir
    if include_test_dir:
        test_dataset = image_dataset_from_directory(
            test_dir,
            label_mode=label_mode,
            image_size=image_size,
            shuffle=False,  # dont need to shuffle
            batch_size=batch_size,
        )
        return train_dataset, valid_dataset, test_dataset, class_names
    else:
        return train_dataset, valid_dataset, class_names


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


def extract_labels_from_dataset(
    files: tf.data.Dataset, is_categorical: bool = True
) -> np.ndarray:
    """
    Extracts true labels from an ImageDataset.

    This function iterates over an unbatched ImageDataset and extracts the true labels
    for each image. If is_categorical is True, it is assumed that the labels are one-hot
    encoded and the function returns the indices of the maximum values (argmax) as the true labels.
    If is_categorical is False, it is assumed that the labels are already provided as integer indices.

    Args:
        files (tf.data.Dataset): The image dataset from which to extract the true labels.
                                 The dataset is expected to tuples of (images, labels).
        is_categorical (bool): Indicates whether the labels are one-hot encoded (True) or
                               not (False).

    Returns:
        np.ndarray: An array of true labels extracted from the dataset.

    Example usage:
        extract_labels_from_dataset(files, is_categorical=True)
    """
    y_true = []
    for images, labels in files.unbatch():
        if is_categorical:
            # One-hot encoded labels: convert to indices
            labels = np.argmax(labels.numpy(), axis=-1)
        else:
            # Labels are already integer indices: directly use them
            labels = labels.numpy()

        y_true.append(labels)

    return np.array(y_true)
