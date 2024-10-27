from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor


def preprocess_audio_datasett():
    dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")

    # upsample rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # using sample
    # audio_input = [dataset[0]["audio"]["array"]]
    # feature_extractor(audio_input, sampling_rate=16000)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            padding=True,
            max_length=100000,
            truncation=True,
        )
        return inputs

    # five sampels
    processed_dataset = preprocess_function(dataset[:5])
    # or complete dataset
    processed_dataset = dataset.map(preprocess_function, batched=True)
    return processed_dataset
