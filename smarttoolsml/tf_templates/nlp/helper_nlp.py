import random

import numpy as np
import pandas as pd
import transformers
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import OneHotEncoder


def print_examples(df: pd.DataFrame, columns: list[str], n_examples: int = 6) -> None:
    """
    Randomly selects and prints examples from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which examples will be selected and printed.
        columns (list[str]): A list of column names to be included in the output.
        n_examples (int, optional): The number of examples to print. Defaults to 6.

    Returns:
        None: This function does not return a value. It prints the selected examples directly to the console.

    Example usage:
        print_examples(df=df_train, columns=["Target", "Text"])
    """
    for i in range(n_examples):
        random_idx = random.choice(df.index)
        values = df.loc[random_idx, columns].tolist()
        print(f"Target: {values[0]}\nText:\n{values[1]}\n\n")


def get_token_counts(
    df_with_column: pd.Series,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_length: int = 512,
    truncation: bool = True,
    padding: bool = False,
) -> np.ndarray:
    """
    Tokenizes text data from a Pandas Series using a specified tokenizer and calculates token lengths for each text entry.
    It prints statistics about the token counts, including the maximum, mean, and minimum lengths,
    and returns an array of individual token lengths. This function is useful for analyzing the distribution of token lengths in your dataset,
    which can inform decisions on sequence length settings for training transformer-based models.

    Args:
        df_with_column (pd.Series): A Pandas Series object containing the text data to be tokenized.
        tokenizer (transformers.PreTrainedTokenizerBase): An instance of a tokenizer from Hugging Face's Transformers library.
        max_length (int, optional): The maximum length of the tokenized sequences.
        truncation (bool, optional): Specifies whether sequences should be truncated to `max_length`. Defaults to True.
        padding (bool, optional): Specifies whether sequences should be padded to `max_length`. This is often not necessary for length analysis and defaults to False.

    Returns:
        np.ndarray: An array containing the token lengths for each text entry in the input Series.
                    This allows for further analysis or visualization of token length distribution.

    Example usage:
        text_series = df["Text"]
        token_lengths = get_token_counts(df_with_column=text_series,
                                         tokenizer=tokenizer,
                                         max_length=512)
    """

    token_lens = []

    for text in df_with_column.values:
        tokens = tokenizer.encode(
            text, max_length=max_length, truncation=truncation, padding=padding
        )
        token_lens.append(len(tokens))

    print(f"Max Token Len: {np.max(token_lens)}")
    print(f"Mean Token Len: {np.mean(token_lens)}")
    print(f"Min Token Len: {np.min(token_lens)}")

    return np.array(token_lens)


def random_over_sampler(
    df_with_text_column: pd.Series, df_with_target_column: pd.Series
) -> pd.DataFrame:
    """
    Applies random over-sampling to balance the distribution of classes in a dataset.
    This function takes separate Series for text data and target labels, oversamples instances in underrepresented classes,
    and returns a new DataFrame with the balanced dataset.

    Args:
        df_with_text_column (pd.Series): A Pandas Series containing text data. Each entry in this series corresponds to one text instance.
        df_with_target_column (pd.Series): A Pandas Series containing the target labels associated with the text data.

    Returns:
        pd.DataFrame: A DataFrame with two columns, "Text" and "Target", representing the balanced dataset after applying random over-sampling.

    Example usage:
        balanced_df = random_over_sampler(df["Text"], df["Target"])
        balanced_df.head()

    Note:
        This function uses the `RandomOverSampler` class from the `imblearn.over_sampling` module to perform the over-sampling.
    """
    ros = RandomOverSampler()

    x_text = np.array(df_with_text_column).reshape(-1, 1)
    y_target = np.array(df_with_target_column).reshape(-1, 1)

    text, target = ros.fit_resample(x_text, y_target)

    data = list(zip([x[0] for x in text], target))
    df = pd.DataFrame(data, columns=["Text", "Target"])

    return df


def one_hot_encode_labels(labels: list[int]) -> np.ndarray:
    """
    Converts a list of numerical labels into a one-hot encoded format.

    Args:
        labels (list[int]): A list of integer labels to be one-hot encoded.

    Returns:
        np.ndarray: A NumPy array of shape (n_samples, n_classes) containing the one-hot encoded labels.

    Example usage:
        labels = [0, 1, 2, 1]
        encoded_labels = one_hot_encode_labels(labels)

    Note:
        This function utilizes the `OneHotEncoder` class from scikit-learn's preprocessing module.
    """
    ohe = OneHotEncoder()
    train = ohe.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    return train
