from aerosound.ml_logic.preprocessing import preprocess_audio
import pandas as pd
import numpy as np

def build_metadata_df(audio_paths):
    """Remove the prefix from a list of audio paths, split the paths into metadata
    fields, and return them as a DataFrame.

    Parameters
    -----------
    audio_paths : list
        List of audio paths.

    Returns
    --------
    df : pd.DataFrame
        DataFrame with 'noise', 'machine', 'id', 'label', and 'file' metadata columns.
    """
    relative_paths = [path.removeprefix('kaggle_dataset_') for path in audio_paths]

    path_parts = [path.split('_') for path in relative_paths]

    metadata = []

    for path in path_parts:

        noise = "_".join(path[:3])

        machine_id = "_".join(path[4:6])

        metadata_row = [noise, path[3], machine_id, path[-2], path[-1]]

        metadata.append(metadata_row)

    df = pd.DataFrame(
        metadata,
        columns=['noise', 'machine', 'id', 'label', 'file']
        )

    df['snr'] = df['noise'].str.split('_').str[0].astype(int)

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
