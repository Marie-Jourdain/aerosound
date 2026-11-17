import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import os

from pathlib import Path
#from aerosound.params import *
from keras import Sequential, layers
from keras.metrics import Recall
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import inspect

from aerosound.ml_logic.data import *
from aerosound.ml_logic.preprocessing import *

def get_path (path):
    ''' input the path to classification csv. The function will return a list inlucing all wav file paths '''
    df_path = pd.read_csv(path)
    df_path = df_path[df_path['machine'] == 'fan']
    root = Path("raw_data")
    audio_paths = [
        root / row.noise / row.id / row.label / row.file
        for row in df_path.itertuples(index=False)
    ]
    return audio_paths

def get_data_rnn(audio_paths):
    ''' input the path list. The function will return the dataframe contain std & zero crossing rate of all wav files '''
    df = build_dataset(audio_paths)
    return df

#run to get all csv files:
if __name__ == "__main__":
    path_for_train = 'aerosound/model/df_train.csv'
    path_for_val = 'aerosound/model/df_validate.csv'
    path_for_test = 'aerosound/model/df_test.csv'

    path_list_train = get_path(path_for_train)
    path_list_val = get_path(path_for_val)
    path_list_test = get_path(path_for_test)

    df_fan_train = get_data_rnn(path_list_train)
    df_fan_val = get_data_rnn(path_list_val)
    df_fan_test = get_data_rnn(path_list_test)

    df_fan_train.to_csv("aerosound/model/fan_all_train_stdzcr_100.csv", index=False)
    df_fan_val.to_csv("aerosound/model/fan_all_val_stdzcr_100.csv", index=False)
    df_fan_test.to_csv("aerosound/model/fan_all_test_stdzcr_100.csv", index=False)
