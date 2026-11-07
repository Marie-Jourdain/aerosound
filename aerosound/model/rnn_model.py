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
#from aerosound.params import *
from keras import Sequential, layers
from keras.metrics import Recall
from keras.callbacks import EarlyStopping

def data_obtain_rnn(path):
# obtain data from the path and saperated the data by machine type and save the dataframes into a dictionary and return to this dictionary
    data_ori = pd.read_csv(path)
    machine_dfs = {}

    for machine in data_ori['machine'].unique():
        machine_dfs[machine] = data_ori[
            data_ori['machine'] == machine
        ].copy()

    print(machine_dfs.keys())
    return machine_dfs

def prep_data_rnn(data_ori):
    data = data_ori.copy()
    data = data.drop_duplicates()
    data['label'] = data['label'].map({
        'normal': 0,
        'abnormal': 1
    })
    data = data.drop(columns=['noise','machine','id','file','snr','file_num'])
    data = data.rename(columns={'label':'target'})
    X = data.drop(columns = 'target')
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )
    X_train_rnn = X_train.to_numpy().reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )
    X_train_rnn = X_train_rnn * 100

    # X_test_rnn = X_test.to_numpy().reshape(
    #     X_test.shape[0],
    #     X_test.shape[1],
    #     1
    # )


    return X_train_rnn, X_test, y_train, y_test

def rnn_LSTM(X_train_rnn, y_train):
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
        optimizer = 'adam',
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
        patience=5,
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
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weights
    )
    return model, history

def predict_sound(model, X_pred):
    ''' predict normal/abnormal '''
    X_pred_rnn = X_pred.to_numpy().reshape(
        X_pred.shape[0],
        X_pred.shape[1],
        1
    )
    y = model.predict(X_pred_rnn)
    return (y >= 0.5).astype(int)

def model_evaluate(model, X_test, y_test):
    ''' show the performance of the model '''
    y_pred = predict_sound(model, X_test).ravel()
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[0,1],
        display_labels=["Normal", "Abnormal"]
    )

    plt.title("Confusion Matrix")
    plt.show()


# test functions above
if __name__ == "__main__":
    machine_dfs = data_obtain_rnn('../../features_0dB.csv')
    data_ori = machine_dfs['fan']
    X_train_rnn, X_test, y_train, y_test = prep_data_rnn(data_ori)
    rnn_model, history = rnn_LSTM(X_train_rnn, y_train)
    model_evaluate(rnn_model, X_test, y_test)
