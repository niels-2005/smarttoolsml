import evaluate
import numpy as np
import tensorflow as tf
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          TFAutoModelForSequenceClassification, Trainer,
                          TrainingArguments, create_optimizer)
from transformers.keras_callbacks import KerasMetricCallback


def text_classification_pytorch():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    # datacollator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Prediction
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    model.config.id2label[predicted_class_id]


def text_classification_tensorflow():
    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    tf_train_set = model.prepare_tf_dataset(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_imdb["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(
        init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps
    )

    model.compile(optimizer=optimizer)

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics, eval_dataset=tf_validation_set
    )

    model.fit(
        x=tf_train_set,
        validation_data=tf_validation_set,
        epochs=3,
        callbacks=[metric_callback],
    )

    # Prediction
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(**inputs).logits
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    model.config.id2label[predicted_class_id]
