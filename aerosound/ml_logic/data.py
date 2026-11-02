import pandas as pd

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
