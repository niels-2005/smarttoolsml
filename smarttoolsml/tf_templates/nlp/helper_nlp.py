import pandas as pd 
import numpy as np
import random 
import re, string 
import demoji
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


def remove_emojis(text: str) -> str:
    """
    Removes emojis from the input text using the `demoji` library.

    Args:
        text (str): The input text from the emojis needs to be removed.

    Returns:
        str: The input text with all emojis removed.
    """
    return demoji.replace(text, "")


def strip_all_entities(text: str) -> str:
    """
    Cleans the input text by removing new line characters, converting to lowercase,
    and stripping out URLs, mentions, non-ASCII characters, and certain punctuation marks.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text, with specified characters and patterns removed.
    """
    text = text.replace("\r", "").replace("\n", " ").replace("\n", " ").lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    banned_list = string.punctuation + "Ã" + "±" + "ã" + "¼" + "â" + "»" + "§"
    table = str.maketrans("", "", banned_list)
    text = text.translate(table)
    return text


def clean_hashtags(tweet: str) -> str:
    """
    Processes hashtags in a tweet by removing hashtags at the end of the tweet and stripping the hashtag symbol from those in the middle.

    Args:
        tweet (str): The tweet text to be processed.

    Returns:
    str: The modified tweet text, where middle-positioned hashtags have been converted into plain words by removing the '#' symbol,
         and any hashtags at the tweet's end have been deleted.
    """
    new_tweet = " ".join(
        word.strip()
        for word in re.split("#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)", tweet)
    )
    new_tweet2 = " ".join(word.strip() for word in re.split("#|_", new_tweet))
    return new_tweet2


def filter_chars(a: str) -> str:
    """
    Filters out words containing special characters such as '&' and '$' from the input string.

    Args:
        a (str): The input string from which words containing '&' and '$' should be removed.

    Returns:
        str: The cleaned string with words containing the specified special characters removed.
    """
    sent = []
    for word in a.split(" "):
        if ("$" in word) | ("&" in word):
            sent.append("")
        else:
            sent.append(word)
    return " ".join(sent)


def remove_mult_spaces(text: str) -> str:
    """
    Reduces multiple consecutive spaces in the input text to a single space.

    Args:
        text (str): The input text with potentially multiple consecutive spaces.

    Returns:
        str: The input text with all multiple spaces reduced to single spaces.
    """
    return re.sub("\s\s+", " ", text)


def get_cleaned_texts(df_with_column: pd.Series) -> np.ndarray:
    """
    This cleaning pipeline includes removing emojis, stripping all entities (like URLs and mentions), cleaning hashtags, 
    filtering out special characters, and finally removing multiple spaces. The result is a NumPy array of cleaned text strings.

    Args:
        df_with_column (pd.Series): A Pandas Series object containing the text data to be cleaned.

    Returns:
        np.ndarray: A NumPy array containing the cleaned text strings.

    Example usage:
        text_series = df["Text"]
        texts_clean(df_with_column=text_series)
    """
    texts_clean = []
    for text in df_with_column.values:
        texts_clean.append(
            remove_mult_spaces(
                filter_chars(clean_hashtags(strip_all_entities(remove_emojis(text))))
            )
        )
    return np.array(texts_clean)


def get_word_counts(df_with_column: pd.Series) -> np.ndarray:
    """
    Calculates the word counts of each text entry in a Pandas Series and returns these lengths in a NumPy array. 
    
    Args:
        df_with_column (pd.Series): A Pandas Series object containing the text data for which the lengths are to be calculated. 
                                    Each entry in the series is assumed to be a single text string.

    Returns:
        np.ndarray: A NumPy array of integers, where each element represents the word count of the corresponding text string in the input Series.

    Example usage:
        text_series = df["Text"]
        text_lengths = get_text_len(df_with_column=text_series)
    """
    text_len = []
    for text in df_with_column.values:
        length = len(text.split())
        text_len.append(length)
    return np.array(text_len)


def get_token_counts(df_with_column: pd.Series, 
                     tokenizer: transformers.PreTrainedTokenizerBase, 
                     max_length: int = 512, 
                     truncation: bool = True, 
                     padding: bool = False) -> np.ndarray:
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
        tokens = tokenizer.encode(text, max_length=max_length, truncation=truncation, padding=padding)
        token_lens.append(len(tokens))

    print(f"Max Token Len: {np.max(token_lens)}")
    print(f"Mean Token Len: {np.mean(token_lens)}")
    print(f"Min Token Len: {np.min(token_lens)}")
    
    return np.array(token_lens)


def random_over_sampler(df_with_text_column: pd.Series, df_with_target_column: pd.Series) -> pd.DataFrame:
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



