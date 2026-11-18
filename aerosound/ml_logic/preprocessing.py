import librosa
import numpy as np
import os
import tensorflow as tf
from aerosound.params import *


def load_audio(audio_path):
    """Load a multichannel audio file without resampling or converting it to mono.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to load.

    Returns
    -------
    signal : np.ndarray
        Audio time series. Multichannel audio is preserved.
    """

    signal, _ = librosa.load(audio_path, sr=None, mono=False)
    return signal


def convert_to_mono(signal):
    """Convert a multichannel audio time series to mono by averaging the channels.

    Parameters
    ----------
    signal : np.ndarray
        Multichannel audio time series with shape (n_channels, n_samples).

    Returns
    -------
    mono_signal : np.ndarray
        Mono audio time series with shape (n_samples,).
    """

    mono_signal = signal.mean(axis=0)

    return mono_signal

def extract_type_features(mono_signal):
    """Split a mono audio time series into 100 time windows and convert them to features
    representing the standard deviation of the audio amplitude within each time window.

    Parameters
    ----------
    mono_signal : np.ndarray
        Mono audio time series with shape (n_samples, ).

    Returns
    -------
    extracted_features : np.ndarray
        Features from the splitting of the mono signal into time windows representing
        the standard deviation of the audio amplitude within one time window with
        shape (100,).
    """

    split_means = np.array_split(mono_signal, 100)

    slice_type_1 = np.array([np.std(s) for s in split_means])

    slice_type_2 = np.array([np.mean(np.diff(np.signbit(s)) != 0) for s in split_means])

    extracted_features = np.column_stack((slice_type_1, slice_type_2))
    return extracted_features

def preprocess_audio(audio_path):
    """Orchestrate the whole preprocessing steps. Load the audio file to store a
    signal. Convert the signal into mono. Extract 100 standard-deviation features.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to load.

    Returns
    -------
    extracted_features : np.ndarray
        Standard-deviation features extracted from 100 time windows, with shape (100,).
    """

    signal = load_audio(audio_path)

    mono_signal = convert_to_mono(signal)

    extracted_features = extract_type_features(mono_signal)

    return extracted_features


#---------------------------------------
# functions for the yamnet model
#---------------------------------------

def wav_to_embeddings(path, model_yam):

    y, _ = librosa.load(path, sr=16000, mono=True)   # 8 channels -> mean
    _, emb, _ = model_yam(y.astype(np.float32))

    return np.mean(emb, axis=0)

_model_yam = None

def get_yamnet():
    # once it is loaded, the model will be reused in this module, each time a model will be needed
    global _model_yam
    if _model_yam is None:
        yamnet_path = os.path.join(ROOT_PATH, 'models', 'yamnet-emb')
        _model_yam = tf.saved_model.load(yamnet_path)
    return _model_yam

def preprocess_yamnet(path):    # path to the .wav-file or the file itsself

    model_yam = get_yamnet()
    embedding = wav_to_embeddings(path, model_yam)

    return embedding
