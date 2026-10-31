import librosa

def load_audio(audio_path):
    """Load a multichannel audio file without resampling or converting it to mono.

    Parameters

    audio_path : str
        Path to the audio file to load
    Returns

    signal : np.ndarray
        Audio time series. Multichannel audio is preserved
    """

    signal, _ = librosa.load(audio_path, sr=None, mono=False)
    return signal
