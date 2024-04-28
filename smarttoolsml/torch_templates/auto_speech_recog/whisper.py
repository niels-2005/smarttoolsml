import torch
from IPython.display import Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def get_pipeline(model_id: str, low_cpu_mem_usage: bool, use_safetensors: bool):
    """
    Initializes and configures a speech-to-text pipeline using the specified model and settings.

    Args:
        model_id (str): Identifier for the pretrained model to be loaded from Hugging Face Hub.
        low_cpu_mem_usage (bool): If True, reduces CPU memory usage by using optimized settings.
        use_safetensors (bool): If True, utilizes SafeTensors to enhance memory safety during processing.

    Returns:
        pipeline: A configured ASR (Automatic Speech Recognition) pipeline ready for performing speech recognition tasks.

    Example usage:
        asr_pipeline = get_pipeline(
            model_id="openai/whisper-large-v3",
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=use_safetensors,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


def auto_speech_recognition(
    audio_data: Audio,
    language: str = "english",
    model_id: str = "openai/whisper-large-v3",
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
    return_result: bool = False,
):
    """
    Performs automatic speech recognition on the provided audio data using a specified model,
    with an option to translate the recognized text into a specified language.

    Args:
        audio_data (Audio): The audio data to be processed.
        language (str, optional): Target language for translation of the recognized text. Defaults to "english".
        model_id (str, optional): The model identifier for speech recognition and translation. Defaults to "openai/whisper-large-v3".
        low_cpu_mem_usage (bool, optional): If set to True, optimizes CPU memory usage. Defaults to True.
        use_safetensors (bool, optional): If set to True, enhances memory safety during processing. Defaults to True.
        return_result (bool, optional): If True, returns the recognition and translation result instead of printing it. Defaults to False.

    Returns:
        dict or None: Returns the speech recognition and translation result as a dictionary if `return_result` is True, otherwise None.

    Example usage:
        result = auto_speech_recognition(
            audio_data=some_audio_clip,
            language="german",
            return_result=True
        )
    """
    pipe = get_pipeline(
        model_id=model_id,
        low_cpu_mem_usage=low_cpu_mem_usage,
        use_safetensors=use_safetensors,
    )

    result = pipe(audio_data, generate_kwargs={f"language": {language}})
    print(result)

    if return_result:
        return result
