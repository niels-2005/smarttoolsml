from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)


def count_vectorizer(documents: list):
    count = CountVectorizer()
    bag = count.fit_transform(documents)
    print(count.vocabulary_)
    return bag
