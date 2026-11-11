from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import pickle # skipped later

from aerosound.params import *
from aerosound.ml_logic.preprocessing import preprocess_audio
#from aerosound.model.model import data_obtain, prep_data

'''
API that takes a WAV-file as upload, converts it and makes a prediction.
'''

app = FastAPI()

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

@app.post("/predict")
def predict_audio(file: UploadFile):

    # TODO: wrap up in a process_function in interface/main.py ?
    features = preprocess_audio(file.file) # features are a np-array

    features = features.reshape(1,100)

    #data = data_obtain(features)
    print("Features extracted")

    model_path = os.path.join(ROOT_PATH, 'models', 'best_knn_fan.pkl')
    print(model_path)
    if MODEL_TARGET == 'local':
        print(model_path)
        with open(model_path, "rb") as file1:
            my_model = pickle.load(file1)

    print("model loaded")

    prediction = my_model.predict(features)

    print(f'Made the following prediction: {prediction[0]}!')

    return {"class": int(prediction[0])}

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
