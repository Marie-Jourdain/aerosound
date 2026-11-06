from aerosound.ml_logic import data
import numpy as np
import pandas as pd

def test_build_metadata_df():

    audio_paths = ['raw_data/kaggle_dataset_-6_dB_fan_fan_id_00_normal_00000000.wav',
                   'kaggle_dataset_0_dB_pump_pump_id_02_abnormal_00000123.wav']
    expected_columns = ['noise', 'machine', 'id', 'label', 'file', 'snr']

    df = data.build_metadata_df(audio_paths)

    assert df.columns.tolist() == expected_columns
    assert df.loc[0, 'noise'] == '-6_dB_fan'
    assert df.loc[0, 'machine'] == 'fan'
    assert df.loc[0, 'id'] == 'id_00'
    assert df.loc[0, 'label'] == 'normal'
    assert df.loc[0, 'file'] == '00000000.wav'
    assert df.loc[1, 'noise'] == '0_dB_pump'
    assert df.loc[1, 'machine'] == 'pump'
    assert df.loc[1, 'id'] == 'id_02'
    assert df.loc[1, 'label'] == 'abnormal'
    assert df.loc[1, 'file'] == '00000123.wav'
    assert df.loc[0, 'snr'] == -6
    assert df.loc[1, 'snr'] == 0

def test_build_features():
    audio_path = 'raw_data/kaggle_dataset_-6_dB_fan_fan_id_00_normal_00000000.wav'
    audio_paths = [audio_path, audio_path]

    extracted_features = data.build_features(audio_paths)

    assert isinstance(extracted_features, np.ndarray)
    assert extracted_features.shape == (2,100)
    assert np.allclose(extracted_features[0], extracted_features[1])


def test_build_dataset():
    audio_path = 'raw_data/kaggle_dataset_-6_dB_fan_fan_id_00_normal_00000000.wav'
    audio_paths = [audio_path, audio_path]

    expected_meta_cols = ['noise', 'machine', 'id', 'label', 'file', 'snr']
    expected_feature_cols = [f'f_{i:03}' for i in range(100)]

    df =  data.build_dataset(audio_paths)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 106)
    assert df.columns[:6].tolist() == expected_meta_cols
    assert df.columns[6:].tolist() == expected_feature_cols
