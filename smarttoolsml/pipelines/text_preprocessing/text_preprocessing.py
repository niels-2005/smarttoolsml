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
    df["clean_text"] = [
        remove_mult_spaces(
            filter_chars(
                remove_stopwords(
                    clean_hashtags(strip_all_entities(remove_emojis(text)))
                )
            )
        )
        for text in df["text"].values
    ]

    print("Getting Word Counts")
    df["text_len"] = [len(text.split()) for text in df["clean_text"].values]

    return df
