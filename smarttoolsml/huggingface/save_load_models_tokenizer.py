from transformers import AutoModelForSequenceClassification, AutoTokenizer


def save_tokenizer_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    save_path = "./models"
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


def load_tokenizer_model():
    save_path = "./models"
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    return model, tokenizer


def load_tokenizer():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
