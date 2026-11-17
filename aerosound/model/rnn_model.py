import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, recall_score
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import os
#from aerosound.params import *
from keras import Sequential, layers
from keras.metrics import Recall
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model

from aerosound.ml_logic.data import *
from aerosound.ml_logic.preprocessing import *

def data_obtain_rnn(path):
    # read the data from csv
    data_ori = pd.read_csv(path)
    return data_ori


def prep_data_rnn(data_ori):
    # clean the data for X and y
    scaler = StandardScaler()
    data = data_ori.copy()
    data = data.drop_duplicates()
    data['label'] = data['label'].map({
        'normal': 0,
        'abnormal': 1
    })
    data = data.drop(columns=['noise','machine','id','file','snr'])
    data = data.rename(columns={'label':'target'})
    X = data.drop(columns = 'target')
    y = data['target']

    X_trans = X.to_numpy().reshape(
        X.shape[0],
        X.shape[1]//2,
        2
    )

    X_trans_scaled = scaler.fit_transform(
        X_trans.reshape(-1, 2)
    ).reshape(X.shape[0], X.shape[1]//2, 2)

    return X_trans_scaled, y

def oversample_data(X_train, y_train):
    # make train data normal/abnormal balance
    y_train = np.asarray(y_train).ravel()

    # Oversample training data
    n_samples, n_steps, n_features = X_train.shape

    X_train_2d = X_train.reshape(n_samples, -1)

    ros = RandomOverSampler(
        sampling_strategy=1.0,
        random_state=42
    )

    X_resampled, y_resampled = ros.fit_resample(
        X_train_2d, y_train
    )

    # Restore 3D shape
    X_resampled = X_resampled.reshape(
        -1, n_steps, n_features
    )

    return X_resampled, y_resampled


def rnn_LSTM(X_train_rnn, X_val, y_train, y_val):
    ''' estublish the model, compile the model, train the model '''
    #estublish the model
    model = Sequential([
        layers.Input(shape=(X_train_rnn.shape[1],X_train_rnn.shape[2])),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    #compile the model
    model.compile(
        optimizer = Adam(learning_rate=0.0001, clipnorm=1.0),
        loss = 'binary_crossentropy',
        metrics=[
            Recall(name='recall'),
            'accuracy',
            'AUC'
        ]
    )

    #define early stop
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True
    )

    #add class weight for abnormal samples, because of unbalanced normal/abnormal data
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=np.asarray(y_train)
    )
    class_weights = dict(zip(classes, weights))

    #train the model
    history = model.fit(
        X_train_rnn,
        y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        class_weight=class_weights
    )
    return model, history

def make_scaler(data_train):
    scaler = StandardScaler()
    data = data_train.copy()
    data = data.drop_duplicates()
    data['label'] = data['label'].map({
        'normal': 0,
        'abnormal': 1
    })
    data = data.drop(columns=['noise','machine','id','file','snr'])
    data = data.rename(columns={'label':'target'})
    X = data.drop(columns = 'target')
    y = data['target']

    scaler.fit(X.to_numpy().reshape(-1, 2))

    return scaler

def predict_API(model, path_wav, scaler):
    ''' via API input wav file and output predict normal/abnormal '''
    # Extract features from WAV
    X = preprocess_audio(path_wav)

    # (100, 2) -> (1, 100, 2)
    X = X.reshape(1, X.shape[0], X.shape[1])

    # Apply the training scaler
    X_scaled = scaler.transform(
        X.reshape(-1, 2)
    ).reshape(X.shape)

    # Predict
    y = model.predict(X_scaled)
    return (y >= 0.4).astype(int).item()

def predict_sound(model, X_pred):
    ''' when test the model with X_test and y_test, run this predict '''
    X_pred_rnn = X_pred.reshape(X_pred.shape[0], X_pred.shape[1], X_pred.shape[2])
    #X_pred_rnn = X_pred_rnn * 100

    y = model.predict(X_pred_rnn)
    return (y >= 0.4).astype(int), y

def model_evaluate(model, X_test, y_test):
    ''' show the performance of the model '''
    y_pred, y_prob_pred = predict_sound(model, X_test)

    y_true = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob_pred = np.asarray(y_prob_pred).ravel()

    auc = roc_auc_score(y_true, y_prob_pred)
    recall = recall_score(y_true, y_pred)

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Recall: {recall:.4f}")

    #print(y_pred)
    #print(y_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred.ravel(),
        labels=[0,1],
        display_labels=["Normal", "Abnormal"]
    )

    plt.title("Confusion Matrix")
    plt.show()


#run the model training:
if __name__ == "__main__":
    #get the data from csv
    data_train = data_obtain_rnn('aerosound/model/fan_all_train_stdzcr_100.csv')
    data_val = data_obtain_rnn('aerosound/model/fan_all_val_stdzcr_100.csv')
    data_test = data_obtain_rnn('aerosound/model/fan_all_test_stdzcr_100.csv')

    #clean the data
    X_train, y_train = prep_data_rnn(data_train)
    X_val, y_val = prep_data_rnn(data_val)
    X_test, y_test = prep_data_rnn(data_test)

    #oversampling the training data
    X_train_resampled, y_train_resampled = oversample_data(X_train, y_train)

    #train the model
    rnn_model, history = rnn_LSTM(X_train_resampled, X_val, y_train_resampled, y_val)
