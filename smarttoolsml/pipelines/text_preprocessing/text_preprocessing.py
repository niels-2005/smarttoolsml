import pandas as pd 
from helper_cleaning import *

def text_preprocessing_pipe(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    print(f"Length Original DataFrame: {len(df)}")
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"Length Updated DataFrame (No Dups, No NaN's): {len(df)}")

    print("Shuffeling DataFrame.")
    df = df.sample(frac=1, random_state=42)

    print("Getting cleaned Texts.")
    df['clean_text'] = [
        remove_mult_spaces(
            filter_chars(
                clean_hashtags(
                    strip_all_entities(
                        remove_emojis(text)
                    )
                )
            )
        ) for text in df["text"].values
    ]

    print("Getting Word Counts")
    df['text_len'] = [len(text.split()) for text in df["clean_text"].values]

    return df


def convert_labels(df: pd.DataFrame, labels_dict: dict):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        labels_dict (dict): _description_

    Returns:
        _type_: _description_

    Example usage:
        labels_dict = {"Ham": 0, "Spam": 1}
        convert_labels(df=df, labels_dict=labels_dict)
    """
    df["label"] = df["label"].map(labels_dict)
    return df


def print_random_texts(df: pd.DataFrame, text_type):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        type (_type_): _description_
    
    Example usage:
        type = "Ham" if text["label"] == 0 else "Spam"
        print_random_texts(df=df, type=type)
    """
    idx = random.randint(0, len(df) - 1)
    text = df.iloc[idx]
    print(f"Label: {text_type}\n\nText: {text['text']}\n\n\n")