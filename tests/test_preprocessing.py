from aerosound.ml_logic import preprocessing
import numpy as np

def test_load_audio():

    # Integration test using a reference MIMII audio file stored locally in raw_data/

    audio_path = 'raw_data/kaggle_dataset_-6_dB_fan_fan_id_00_normal_00000000.wav'
    signal = preprocessing.load_audio(audio_path)
    assert signal.shape == (8, 160000)
    assert isinstance(signal, np.ndarray)


def test_convert_to_mono():

    signal = np.array([
        [1, 2, 3],
        [3, 4, 5]
    ])
    expected_mono_signal = np.array([2., 3., 4.])

    mono_signal = preprocessing.convert_to_mono(signal)

    assert np.array_equal(mono_signal, expected_mono_signal)
    assert mono_signal.shape == (3,)
