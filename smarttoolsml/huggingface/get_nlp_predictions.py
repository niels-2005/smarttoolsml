import tensorflow as tf
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)


def get_pt_predictions():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pt_batch = tokenizer(
        [
            "We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it.",
        ],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    pt_outputs = pt_model(**pt_batch)

    pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    print(pt_predictions)


def get_tf_predictions():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tf_batch = tokenizer(
        [
            "We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it.",
        ],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="tf",
    )
    tf_outputs = tf_model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
    print(tf_predictions)
