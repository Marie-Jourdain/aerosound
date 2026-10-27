from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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



@app.get("/predict")
def predict(wav_name: str): # filepath to the wav

    # "/kaggle/input/datasets/bisheshgiri/mimii-dataset/-6_dB_fan/fan/id_00/abnormal/00000000.wav"

    short = wav_name.replace("/kaggle/input/datasets/bisheshgiri/mimii-dataset/",'')
    parts=wav_name.split(sep='/')
    classification = parts[-2]

    return {
        'predicted classification':
            classification
    }
