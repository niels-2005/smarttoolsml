import evaluate
import pandas as pd
from datasets import load_dataset
from evaluate import evaluator
from evaluate.visualization import radar_plot
from transformers import pipeline


def evaluate_accuracy():
    accuracy = evaluate.load("accuracy")
    # accuracy.description if needed
    # accuracy.features

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 1]

    accuracy.compute(references=y_true, predictions=y_pred)


def evaluate_more_metrics():
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 1]

    clf_metrics.compute(references=y_true, predictions=y_pred)


def evaluate_with_task_evaluator():
    pipe = pipeline("text-classification", model="lvwerra/distilbert-imdb", device=0)
    data = load_dataset("imdb", split="test").shuffle().select(range(1000))
    metric = evaluate.load("accuracy")

    task_evaluator = evaluator("text-classification")
    results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=data,
        metric=metric,
        label_mapping={"NEGATIVE": 0, "POSITIVE": 1},
    )
    print(results)


def evaluate_several_models():
    models = [
        "xlm-roberta-large-finetuned-conll03-english",
        "dbmdz/bert-large-cased-finetuned-conll03-english",
        "elastic/distilbert-base-uncased-finetuned-conll03-english",
        "dbmdz/electra-large-discriminator-finetuned-conll03-english",
        "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner",
        "philschmid/distilroberta-base-ner-conll2003",
        "Jorgeutd/albert-base-v2-finetuned-ner",
    ]

    data = load_dataset("conll2003", split="test").shuffle().select(1000)
    task_evaluator = evaluator("token-classification")

    results = []
    for model in models:
        results.append(
            task_evaluator.compute(model_or_pipeline=model, data=data, metric="seqeval")
        )

    df = pd.DataFrame(results, index=models)
    df[
        [
            "overall_f1",
            "overall_accuracy",
            "total_time_in_seconds",
            "samples_per_second",
            "latency_in_seconds",
        ]
    ]


def plot_results(results, model_names):
    plot = radar_plot(
        data=results, model_names=model_names, invert_range=["latency_in_seconds"]
    )
    plot.show()
