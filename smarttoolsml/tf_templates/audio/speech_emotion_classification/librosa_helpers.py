import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt


def load_audio_file(path):
    """_summary_

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_

    Example usage:
        path = /*/*.wav
        data, sr = load_audio_file(path)
    """
    data, sr = librosa.load(path)
    return data, sr


def play_and_plot_audio(path):
    data, sr = librosa.load(path)
    plt.figure(figsize=(12, 6))
    librosa.display.waveshow(y=data, sr=sr)
    ipd.Audio(data, rate=sr)
