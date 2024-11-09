import datasets
import pandas as pd
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def download_dataset_shuffle_n_samples():
    dataset = datasets.load("glue", "sst2", split="test").shuffle().select(range(1000))
    return dataset


def download_dataset():
    dataset = datasets.load("glue", "sst2")
    return dataset


def download_dataset_single_split():
    dataset = datasets.load("glue", "sst2", split="train")
    return dataset


def download_dataset_pandas_dataframe():
    dataset = datasets.load("glue", "sst2")
    df = pd.DataFrame(dataset)
    return df


def datasets_ready_for_torch():
    dataset = datasets.load("glue", "sst2")
    dataset.set_format(type="torch", columns=["input_values", "labels"])
    dataloader = DataLoader(dataset, batch_size=4)
    return dataloader


def datasets_ready_for_tf():
    dataset = datasets.load("glue", "sst2")
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    tf_dataset = model.prepare_tf_dataset(
        dataset,
        batch_size=4,
        shuffle=True,
    )
    return tf_dataset


def download_dataset_and_train_test_split():
    squad = datasets.load("squad", split="train[:5000]")
    squad = squad.train_test_split(test_size=0.2)


def dataset_from_pandas(df_train_small, df_validation_small):
    train_dataset = Dataset.from_pandas(df_train_small)
    validation_dataset = Dataset.from_pandas(df_validation_small)

    dataset = DatasetDict({"train": train_dataset, "validation": validation_dataset})
