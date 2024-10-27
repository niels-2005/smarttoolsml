from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


def tokenize_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_dataset("rotten_tomatoes", split="train")

    def tokenization(example):
        return tokenizer(example["text"], return_tensors="pt")

    dataset = dataset.map(tokenization, batched=True)

    # TESTING NEEDED:
    # dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)

    # dataset for pytorch:
    # dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

    # dataset for tensorflow:
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    # tf_dataset = dataset.to_tf_dataset(
    #     columns=["input_ids", "token_type_ids", "attention_mask"],
    #     label_cols=["label"],
    #     batch_size=2,
    #     collate_fn=data_collator,
    #     shuffle=True
    # )
