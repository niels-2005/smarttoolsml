import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)


def load_pretrained_model(model_name: str, article: str, task: str = "summarization"):
    """_summary_

    Args:
        model_name (str): _description_
        article (str): _description_
        task (str, optional): _description_. Defaults to "summarization".

    Example usage:
        ARTICLE = New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
        checkpoint = "google-t5/t5-small"
        load_pretrained_model(model_name=checkpoint, article=ARTICLE)
    """
    summarizer = pipeline(task, model=model_name)
    summarized_text = summarizer(
        article, max_length=512, min_length=30, do_sample=False
    )
    print(summarized_text[0]["summary_text"])


def preprocess_function(examples):
    """_summary_

    Args:
        examples (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

        prefix = "summarize: "

        # needs to be defined
    """
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_dataset(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        from datasets import load_dataset
        dataset = load_dataset("mlsum", "de", trust_remote_code=True)
        tokenized_dataset = preprocess_dataset(dataset)
    """
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset


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


def load_metric():
    rouge = evaluate.load("rouge")
    return rouge


def load_datgacollator(tokenizer):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model="google-t5/t5-small"
    )
    return data_collator


def finetune_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    training_args = Seq2SeqTrainingArguments(
        output_dir="summarized_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        gradient_accumulation_steps=2,  # if gpu memory isnt enough
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
