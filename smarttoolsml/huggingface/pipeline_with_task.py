from transformers import pipeline


def predictions_with_task():
    """task examples: https://huggingface.co/tasks"""
    classifier = pipeline("sentiment_analysis", device=0)

    results = classifier("I'm so happy tdoay!")
    print(f"{results[0]['label']} with score {results[0]['score']}")


def predictions_with_specific_model():
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
        device=0,
    )

    results = classifier(
        [
            "We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it.",
        ]
    )
    for result in results:
        print(f"{result['label']} with score {result['score']}")
