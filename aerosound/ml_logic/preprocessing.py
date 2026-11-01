import librosa
import numpy as np

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

def extract_std_features(mono_signal):
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

    slice_stds = [s.std() for s in split_means]

    extracted_features = np.array(slice_stds)
    return extracted_features
