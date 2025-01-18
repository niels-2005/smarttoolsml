import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def tokenizer(text):
    return text.split()


## --- ##
stop = stopwords.words("english")


def tokenizer_and_preprocessor(text):
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text.lower())
    text = re.sub("[\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    return [w for w in text.split() if w not in stop]


porter = PorterStemmer()


def tokenizer_porter(text):
    """_summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        tokenizer_porter("runners like running and thus they run")

        it returns: ['runner', 'like', 'run', 'and', 'thu', 'they', 'run']
    """
    return [porter.stem(word) for word in text.split()]
