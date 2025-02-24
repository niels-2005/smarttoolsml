import math

import tensorflow as tf
import torch
from datasets import load_dataset
from transformers import (
    AdamWeightDecay,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TFAutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)


def masked_language_modeling_pytorch():
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    # flatten() is "text" field is nested inside "answers"
    eli5 = eli5.flatten()

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

    training_args = TrainingArguments(
        output_dir="my_awesome_eli5_mlm_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # Prediction
    text = "The Milky Way is a <mask> galaxy."
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    logits = model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

    for token in top_3_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))


def masked_language_modeling_tensorflow():
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")

    # flatten() is "text" field is nested inside "answers"
    eli5 = eli5.flatten()

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf"
    )
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

    model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")

    tf_train_set = model.prepare_tf_dataset(
        lm_dataset["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        lm_dataset["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)  # No loss argument!

    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3)

    # Prediction
    text = "The Milky Way is a <mask> galaxy."
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    logits = model(**inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]

    top_3_tokens = tf.math.top_k(mask_token_logits, 3).indices.numpy()
    for token in top_3_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
