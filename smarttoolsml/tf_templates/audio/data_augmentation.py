import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np


# NOISE
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def test_data_augmentation(path):
    data, sr = librosa.load(path)

    print("Normal Audio:\n")
    ipd.Audio(data, rate=sr)
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(y=data, sr=sr)
    plt.show()

    print("Audio with Noise:\n")
    noise_sound = noise(data)
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(y=noise_sound, sr=sr)
    ipd.Audio(noise_sound, rate=sr)
    plt.show()

    print("Audio with Stretching:\n")
    stretch_sound = stretch(data, rate=0.2)
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(stretch_sound, sr=sr)
    ipd.display(ipd.Audio(stretch_sound, rate=sr))
    plt.show()

    print("Audio with Shifting:\n")
    shifted_sound = shift(data)
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(y=shifted_sound, sr=sr)
    ipd.Audio(shifted_sound, rate=sr)
    plt.show()

    print("Audio with Pitch:\n")
    pitched_audio = pitch(data, sr)
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(y=pitched_audio, sr=sr)
    ipd.Audio(pitched_audio, rate=sr)
    plt.show()
