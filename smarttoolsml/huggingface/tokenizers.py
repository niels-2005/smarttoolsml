from transformers import AutoTokenizer


def encode_decode_sentence_with_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    encoded_input = tokenizer(
        "Do not meddle in the affairs of wizards, for they are subtle and quick to anger."
    )
    print(encoded_input)
    tokenizer.decode(encoded_input["input_ids"])


def encode_batch_with_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]

    encoded_inputs = tokenizer(batch_sentences)
    print(encoded_inputs)


def encode_batch_with_padding_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]

    encoded_inputs = tokenizer(batch_sentences, padding=True)
    print(encoded_inputs)


def encode_batch_with_truncation_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]

    encoded_inputs = tokenizer(batch_sentences, truncation=True)
    print(encoded_inputs)


def encode_batch_with_tokenizer_and_build_tensor():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    batch_sentences = [
        "But what about second breakfast?",
        "Don't think he knows about second breakfast, Pip.",
        "What about elevensies?",
    ]

    # return tensors ["pt", "tf", "np"]
    encoded_input = tokenizer(
        batch_sentences, padding=True, truncation=True, return_tensors="pt"
    )
    print(encoded_input)
