import datasets
from transformers import pipeline


def load_dataset_with_metric_and_evaluate_it():
    dataset = datasets.load("glue", "sst2")
    metric = datasets.load_metric("glue", "sst2")

    n_samples = 500

    X = dataset.data["sentence"].to_pylist()[:n_samples]
    y = dataset.data["label"].to_pylist()[:n_samples]

    classifier = pipeline("sentiment_analysis", device=0)
    results = classifier(X)
    predictions = [0 if res["label"] == "NEGATIVE" else 1 for res in results]

    print(metric.compute(predictions=predictions, references=y))
