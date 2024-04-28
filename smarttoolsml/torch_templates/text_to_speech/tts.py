import torch
from TTS.api import TTS


def generate_speech(
    text: str,
    output_path: str,
    speaker_path: str,
    language: str = "en",
    preset_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
):
    """
    Generates speech audio from the given text using a specified text-to-speech model and speaker voice.

    Args:
        text (str): The text from which to generate speech.
        output_path (str): The file path where the generated speech audio will be saved.
        speaker_path (str): The file path of the sample voice used by the TTS model to generate speech.
        language (str, optional): The language of the speech. Defaults to "en" (English).
        preset_name (str, optional): The preset configuration of the TTS model to use. Defaults to
                                     "tts_models/multilingual/multi-dataset/xtts_v2".

    Example usage:
        generate_speech(text="Hello, world!", output_path="output.wav", speaker_path="your_voice.wav", language="en")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(preset_name).to(device)
    tts.tts_to_file(
        text=text, file_path=output_path, speaker_wav=speaker_path, language=language
    )
