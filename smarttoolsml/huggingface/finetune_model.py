import evaluate
import numpy as np
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def finetune_with_pytorch():
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased", num_labels=5
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # create smaller datasets if needed
    small_train_dataset = (
        tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    )
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # training arguments, evaluate on epoch end
    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # start training
    trainer.train()


def finetune_with_tensorflow():
    dataset = load_dataset("glue", "cola")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "google-bert/bert-base-cased"
    )

    def tokenize_dataset(data):
        # Keys of the returned dictionary will be added to the dataset as columns
        return tokenizer(data["text"])

    dataset = dataset.map(tokenize_dataset)

    tf_dataset = model.prepare_tf_dataset(
        dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer
    )

    # No Loss Argument!, Huggingface Models automatically choose
    model.compile(optimizer=Adam(3e-5))

    model.fit(tf_dataset)
