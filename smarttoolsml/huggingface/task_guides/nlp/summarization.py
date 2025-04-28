import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AdamWeightDecay, AutoModelForSeq2SeqLM,
                          AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          TFAutoModelForSeq2SeqLM)
from transformers.keras_callbacks import KerasMetricCallback

# IMPORTANT: ABSTRACTIVE SUMMARIZATION!


def summarization_pytorch():
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(
            text_target=examples["summary"], max_length=128, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_billsum = billsum.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model="google-t5/t5-small"
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_billsum_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,  # change to bf16=True for XPU
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Prediction
    text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarization_tensorflow():
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    prefix = "summarize: "

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(
            text_target=examples["summary"], max_length=128, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_billsum = billsum.map(preprocess_function, batched=True)

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    model = TFAutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model="google-t5/t5-small", return_tensors="tf"
    )

    tf_train_set = model.prepare_tf_dataset(
        tokenized_billsum["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_billsum["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model.compile(optimizer=optimizer)  # No loss argument!

    metric_callback = KerasMetricCallback(
        metric_fn=compute_metrics, eval_dataset=tf_test_set
    )

    model.fit(
        x=tf_train_set,
        validation_data=tf_test_set,
        epochs=3,
        callbacks=[metric_callback],
    )

    # Prediction
    text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
    inputs = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    tokenizer.decode(outputs[0], skip_special_tokens=True)
