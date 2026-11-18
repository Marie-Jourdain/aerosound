from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import joblib

from tensorflow import keras

from aerosound.params import *
from aerosound.ml_logic.preprocessing import preprocess_yamnet
from aerosound.ml_logic.registry import load_model
from aerosound.params import ROOT_PATH
from aerosound.model.rnn_model import predict_API

'''
API that takes a WAV-file as upload, converts it and makes a prediction.
'''

# warmup of the API (including loading a model)
#-----------------------------------------------
# preload the models and make a first (dummy)-prediction to have everything set
# before the first real prediction
d_path = os.path.join(ROOT_PATH, 'models/dummy/dummy.wav') # dummy sound file

@asynccontextmanager
async def lifespan(app):
    # preprocess a dummy-file. the models will be loaded globally, so that it can
    # be reused for the prediction
    d_features = preprocess_yamnet(d_path).reshape(1,1024)

    dummy_model = load_model('y_model_fan_2d_k12.keras')
    dummy_model.predict(d_features)
    yield


app = FastAPI(lifespan=lifespan)

# Allowing all middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
        'vital sign': "I'm alive!"
    }


@app.post("/predict_rnn")
def predict_rnn(file: UploadFile):

    rnn_path = os.path.join(ROOT_PATH, 'models', 'lstm_stdzcr_allnoise_100_0.85_0.64.keras')
    rnn_model=keras.models.load_model(rnn_path) # load model directly, the
                                                # registry.load_model-function would reuse
                                                # the preloaded global model which is a cnn-model.
    print("model loaded")

    # get the scaler
    sc_path = os.path.join(ROOT_PATH, 'aerosound', 'model', 'standard_scaler.pkl')
    with open(sc_path, "rb") as s_file:
        scaler = joblib.load(s_file)

    prediction = predict_API(rnn_model, file.file, scaler)
    print(f'Made the following prediction: {prediction}')

    return {"class": prediction}


@app.post("/predict_yam")
def predict_yam(file: UploadFile):

    features = preprocess_yamnet(file.file).reshape(1,1024) # features are a np-array
    print("Features extracted")

    cnn_model = load_model('y_model_fan_2d_k12.keras')
    print("model loaded")

    prediction = cnn_model.predict(features)
    pred_class = 0 if prediction[0][0]<0.5 else 1
    probab = (1-float(prediction[0][0])) if prediction[0][0]<0.5 else float(prediction[0][0])

    print(f'Made the following prediction: {pred_class}, with {float(probab)*100:.2f}% probability!')

    return {"class": pred_class, "probability": probab}

@app.get("/predict_text")
def predict_text(wav_name: str): # filepath to the wav

    # "/kaggle/input/datasets/bisheshgiri/mimii-dataset/-6_dB_fan/fan/id_00/abnormal/00000000.wav"
    short = wav_name.replace("/kaggle/input/datasets/bisheshgiri/mimii-dataset/",'')
    parts=wav_name.split(sep='/')
    classification = parts[-2]

    return {
        'predicted classification':
            classification
    }
