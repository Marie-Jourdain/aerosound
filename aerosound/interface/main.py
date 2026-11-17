import numpy as np
import pandas as pd

from pathlib import Path

from aerosound.model.rnn_model import rnn_LSTM, prep_data_rnn, predict_sound, model_evaluate
from aerosound.ml_logic.registry import load_model, mlflow_run, mlflow_transition_model

@mlflow_run
def train():
    '''
    ...
    '''

    #load training data
    #load validation data
    #load model???

    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")

    return val_mae #TODO: choose metrics or return a dictionary


@mlflow_run
def evaluate():
    pass
    # load new data point

    #X=
    #y=

    # load model
    #model = load_model()

    # = model_evaluate(model, X, y)

def predict(X):
    # load model
    model = load_model()
    # make prediction
    y_pred = predict_sound(model, X)

    return y_pred
