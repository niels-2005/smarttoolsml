import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AdamWeightDecay,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TFAutoModelForSeq2SeqLM,
)
from transformers.keras_callbacks import KerasMetricCallback


def translation_pytorch():
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    source_lang = "en"
    target_lang = "fr"
    prefix = "translate English to French: "

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs

    tokenized_books = books.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model="google-t5/t5-small"
    )

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_opus_books_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=True,  # change to bf16=True for XPU
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Prediction
    text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(
        inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
    )
    tokenizer.decode(outputs[0], skip_special_tokens=True)


def translation_tensorflow():
    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    source_lang = "en"
    target_lang = "fr"
    prefix = "translate English to French: "

    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs

    tokenized_books = books.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model="google-t5/t5-small", return_tensors="tf"
    )

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    model = TFAutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    tf_train_set = model.prepare_tf_dataset(
        tokenized_books["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_books["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)  # No loss argument!

    metric = evaluate.load("sacrebleu")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

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
    text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
    inputs = tokenizer(text, return_tensors="tf").input_ids
    outputs = model.generate(
        inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
    )
    tokenizer.decode(outputs[0], skip_special_tokens=True)
