from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor, Trainer, TrainingArguments


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"][0]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )

        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def asr_pytorch():
    minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
    minds = minds.train_test_split(test_size=0.2)
    minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

    # sampling_rate needs to be 16000
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

    def uppercase(example):
        return {"transcription": example["transcription"].upper()}

    minds = minds.map(uppercase)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=batch["transcription"],
        )
        batch["input_length"] = len(batch["input_values"][0])
        return batch

    encoded_minds = minds.map(
        prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    training_args = TrainingArguments(
        output_dir="my_awesome_asr_mind_model",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=2000,
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_minds["train"],
        eval_dataset=encoded_minds["test"],
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Prediction
    inputs = processor(
        minds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    transcription
