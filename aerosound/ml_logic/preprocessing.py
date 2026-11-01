import librosa


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
