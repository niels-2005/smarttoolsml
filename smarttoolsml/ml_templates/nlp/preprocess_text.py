import re


def preprocessor(text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        preprocessor("</a>This :) is :( a test :-)!")

        it returns: 'this is a test :) :( :)'
    """
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return text


def apply_preprocessor(df, col_name):
    """_summary_

    Args:
        df (_type_): _description_
        col_name (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        df = pd.read_csv(...)
        col_name = "review"
        apply_preprocessor(df, col_name)
    """
    df[col_name] = df[col_name].apply(preprocessor)
    return df
