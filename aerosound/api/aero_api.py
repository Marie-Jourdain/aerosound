from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import pickle # skipped later

from aerosound.params import *
from aerosound.ml_logic.preprocessing import preprocess_audio, preprocess_yamnet
from aerosound.ml_logic.registry import load_model
from aerosound.params import ROOT_PATH
#from aerosound.model.model import data_obtain, prep_data

'''
API that takes a WAV-file as upload, converts it and makes a prediction.
'''
# warmup of the API (including loading a model)

def run_prediction(raw_input):
    x = preprocess(raw_input)
    emb = models["cnn"].predict(x)
    return models["dense"].predict(emb)

d_path = os.path.join(ROOT_PATH, 'models/dummy/dummy.wav')

@asynccontextmanager
async def lifespan(app):
    d_features = preprocess_yamnet(d_path) # features are a np-array
    d_features = d_features.reshape(1,1024)
    dummy_model = load_model('y_model_fan_2d_k12.keras'))
    dummy_model.predict(d_features)
    yield


app = FastAPI(lifespan=lifespan)

# Allowing all middleware is optional, but good practice for dev purposes
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

import time, logging
@app.post("/predict")
def predict_rnn(file: UploadFile):

    features = preprocess_audio(file.file) # features are a np-array

    features = features.reshape(1,100)

    #data = data_obtain(features)
    print("Features extracted")

    #model_path = os.path.join(ROOT_PATH, 'models', 'best_knn_fan.pkl') #TODO: delete if really not needed

    my_model=load_model('20261111-172416.keras')

    print("model loaded")

    prediction = my_model.predict(features)

    print(f'Made the following prediction: {prediction[0]}!')

    return {"class": int(prediction[0])}

@app.post("/predict_yam")
def predict_yam(file: UploadFile):
    t0=time.perf_counter()

    features = preprocess_yamnet(file.file) # features are a np-array

    features = features.reshape(1,1024)

    print("Features extracted")

    my_model = load_model('y_model_fan_2d_k12.keras')

    print("model loaded")

    prediction = my_model.predict(features)
    pred_class = 0 if prediction[0][0]<0.5 else 1
    probab = (1-float(prediction[0][0])) if prediction[0][0]<0.5 else float(prediction[0][0])

    print(f'Made the following prediction: {pred_class}, with {float(probab)*100:.2f}% probability!')
    logging.warning(f"predict took {time.perf_counter() - t0:.2f}s")
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
