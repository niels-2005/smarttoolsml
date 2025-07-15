import evaluate
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)


def audio_classification_pytorch():
    minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
    minds = minds.train_test_split(test_size=0.2)

    # remove unimportant columns
    minds = minds.remove_columns(
        ["path", "transcription", "english_transcription", "lang_id"]
    )

    labels = minds["train"].features["intent_class"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # example label: id2label[str(2)]

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # sampling_rate must be 16000
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=16000,
            truncation=True,
        )
        return inputs

    encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_minds = encoded_minds.rename_column("intent_class", "label")

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir="my_awesome_mind_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Prediction
    inputs = feature_extractor(
        minds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_ids = torch.argmax(logits).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    predicted_label
