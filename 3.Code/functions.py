import pandas as pd
import librosa
import numpy as np


def load_sound(p):
    y, sr = librosa.load(p, sr=None)
    return y


def chromagram(y, sr):
    spec = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=spec, sr=sr).T, axis=0)
    chroma_f = np.array(
        (np.amin(chroma), np.amax(chroma), np.mean(chroma), np.std(chroma))
    )
    return chroma_f


def melspec(y, sr):
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    mel_f = np.array((np.amin(mel), np.amax(mel), np.mean(mel), np.std(mel)))
    return mel_f


def mfcc(y, sr):
    mfc = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)
    mfc_f = np.array((np.amin(mfc), np.amax(mfc), np.mean(mfc), np.std(mfc)))
    return mfc_f


def centroid(y, sr):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_f = np.array((np.amin(cent), np.amax(cent), np.mean(cent), np.std(cent)))
    return cent_f


def onset_number(y):
    onset = librosa.onset.onset_detect(y=y, sr=22050, units="time")
    return onset.shape[0]


def amplitude_envelope(y):
    frame_size = 1024
    hop_length = 512
    ampl = np.array([max(y[i : i + frame_size]) for i in range(0, len(y), hop_length)])
    return np.array((np.amin(ampl), np.amax(ampl), np.mean(ampl), np.std(ampl)))


def rms(y):
    root = librosa.feature.rms(y)
    return np.array((np.amin(root), np.amax(root), np.mean(root), np.std(root)))


def zcr(y):
    zero = librosa.feature.zero_crossing_rate(y)
    return np.array((np.amin(zero), np.amax(zero), np.mean(zero), np.std(zero)))


def get_features(y):
    sr = 160000
    chroma = chromagram(y, sr=sr)
    mel = melspec(y, sr=sr)
    mfc_coef = mfcc(y, sr=sr)
    cetr = centroid(y, sr=sr)
    onst = onset_number(y)
    ampl = amplitude_envelope(y)
    root = rms(y)
    zero = zcr(y)
    feature_matrix = np.array([])
    feature_matrix = np.hstack((chroma, mel, mfc_coef, cetr, onst, ampl, root, zero))
    return (pd.Dataframe(feature_matrix)).T
