from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model_with_auto_classes():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    encoding = tokenizer(
        ["Hello!", "How are you?"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    outputs = model(**encoding)

    pt_predictions = nn.functional.softmax(outputs.logits, dim=-1)
    print(pt_predictions)
