from aerosound.ml_logic.preprocessing import preprocess_audio
import pandas as pd
import numpy as np
from pathlib import Path

# def build_metadata_df(audio_paths):
#     """Remove the prefix from a list of audio paths, split the paths into metadata
#     fields, and return them as a DataFrame.

#     Parameters
#     -----------
#     audio_paths : list
#         List of audio paths.

#     Returns
#     --------
#     df : pd.DataFrame
#         DataFrame with 'noise', 'machine', 'id', 'label', and 'file' metadata columns.
#     """
#     relative_paths = [
#         Path(path).name.removeprefix('kaggle_dataset_')
#         for path in audio_paths
#         ]

#     path_parts = [path.split('_') for path in relative_paths]

#     metadata = []

#     for path in path_parts:

#         noise = "_".join(path[:3])

#         machine_id = "_".join(path[4:6])

#         metadata_row = [noise, path[3], machine_id, path[-2], path[-1]]

#         metadata.append(metadata_row)

#     df = pd.DataFrame(
#         metadata,
#         columns=['noise', 'machine', 'id', 'label', 'file']
#         )

#     df['snr'] = df['noise'].str.split('_').str[0].astype(int)

#     return df

def build_metadata_df(audio_paths):
    """
    made by shaotong, just for testing
    Extract metadata from audio file paths.

    Expected path structure:
    raw_data/0db_fan/id_00/normal/00000005.wav

    Parameters
    ----------
    audio_paths : list
        List of audio file paths.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing:
        noise, machine, id, label, file, snr
    """

    metadata = []

    for audio_path in audio_paths:
        path = Path(audio_path)

        # Extract metadata from folder names
        label = path.parent.name
        machine_id = path.parent.parent.name
        dataset_folder = path.parent.parent.parent.name

        # Extract machine type and SNR
        snr_str, machine = dataset_folder.split("_", 1)
        snr = int(snr_str.lower().replace("db", ""))

        # Build metadata row
        metadata_row = {
            "noise": f"{snr}_dB",
            "machine": machine,
            "id": machine_id,
            "label": label,
            "file": path.name,
            "snr": snr
        }

        metadata.append(metadata_row)

    df = pd.DataFrame(
        metadata,
        columns=["noise", "machine", "id", "label", "file", "snr"]
    )

    return df



def build_features(audio_paths):
    """Build 100 features by applying audio preprocessing to each audio file.

    Parameters
    ----------
    audio_paths : list
        List of audio paths.

    Returns
    -------
    extracted_features : np.ndarray
        Feature matrix containing 100 standard-deviation features per audio file,
        with shape (n_files,100).
    """
    extracted_features = []

    for path in audio_paths:
        features_row = preprocess_audio(path)
        extracted_features.append(features_row)

    extracted_features = np.array(extracted_features)

    return extracted_features

def build_dataset(audio_paths):
    """Build a dataset by combining audio metadata and extracted features into
    a single DataFrame.

    Parameters
    ----------
    audio_paths : list
        List of audio paths.

    Returns
    --------
    metadata_features_df : pd.DataFrame
        DataFrame containing six metadata columns and 100 standard-deviation
        feature columns, with shape (n_files, 106).
    """

    metadata_df = build_metadata_df(audio_paths)

    features = build_features(audio_paths)

    feature_cols = [f"f_{i:03d}" for i in range(400)]

    features_df = pd.DataFrame(features, columns=feature_cols)

    metadata_features_df = pd.concat([metadata_df, features_df], axis=1)

    return metadata_features_df
