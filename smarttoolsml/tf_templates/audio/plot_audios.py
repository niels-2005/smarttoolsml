from glob import glob
from itertools import cycle

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_audio_files():
    audio_files = glob("./audio_speech_actors_01-24/*/*.wav")
    return audio_files


def play_audio_sample(audio_files: list):
    ipd.Audio(audio_files[0])


def plot_samples(
    audio_files: list,
    plot_audio_file: bool = True,
    plot_trimmed: bool = True,
    plot_zoomed: bool = True,
    plot_spectogram: bool = True,
    plot_mel_spectogram: bool = True,
):
    """_summary_

    Args:
        audio_files (list): _description_
        plot_audio_file (bool, optional): _description_. Defaults to True.
        plot_trimmed (bool, optional): _description_. Defaults to True.
        plot_zoomed (bool, optional): _description_. Defaults to True.
        plot_spectogram (bool, optional): _description_. Defaults to True.
        plot_mel_spectogram (bool, optional): _description_. Defaults to True.

    Example usage:
        audio_files = find_audio_files()
        plot_samples(audio_files=audio_files)
    """
    y, sr = librosa.load(audio_files[0])
    print(f"y: {y[:10]}")
    print(f"shape y: {y.shape}")
    print(f"sr: {sr}")

    if plot_audio_file:
        pd.Series(y).plot(figsize=(10, 5), lw=1, title="Raw Audio Example")
        plt.show()

    if plot_trimmed:
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        pd.Series(y_trimmed).plot(
            figsize=(10, 5), lw=1, title="Raw Audio Trimmed Example"
        )
        plt.show()

    if plot_zoomed:
        pd.Series(y[30000:30500]).plot(
            figsize=(10, 5), lw=1, title="Raw Audio Zoomed In Example"
        )
        plt.show()

    if plot_spectogram:
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        print(f"Shape: {S_db.shape}")

        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(S_db, x_axis="time", y_axis="log", ax=ax)
        ax.set_title("Spectogram Example", fontsize=20)
        fig.colorbar(img, ax=ax, format=f"%0.2f")
        plt.show()

    if plot_mel_spectogram:
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128 * 2,
        )
        S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 5))
        img = librosa.display.specshow(S_db_mel, x_axis="time", y_axis="log", ax=ax)
        ax.set_title("Mel Spectogram Example", fontsize=20)
        fig.colorbar(img, ax=ax, format=f"%0.2f")
        plt.show()
