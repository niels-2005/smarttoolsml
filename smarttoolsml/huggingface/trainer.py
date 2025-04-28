from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          TFAutoModelForSequenceClassification, Trainer,
                          TrainingArguments)


def pytorch_trainer():
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    dataset = load_dataset("rotten_tomatoes")

    training_args = TrainingArguments(
        output_dir="path/to/save/folder/",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
    )

    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])

    dataset = dataset.map(tokenize_dataset, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # for tasks like translation or summarization, that use a sequence-to-sequence model, use the seq2seqtrainer and
    # seq2seqtrainingarguments instead
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


def tensorflow_trainer():
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )
    dataset = load_dataset("rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])

    dataset = dataset.map(tokenize_dataset)
    tf_dataset = model.prepare_tf_dataset(
        dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
    )

    model.compile(optimizer="adam")  # No loss argument!
    model.fit(tf_dataset)
