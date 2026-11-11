import os
import time
import glob
from colorama import Fore, Style
import pickle
from aerosound.params import *

from tensorflow import keras
from google.cloud import storage

#import mlflow
#from mlflow.tracking import MlflowClient

#TODO: if we will have different models, code could be inserted to choose model-type when loading and saving

def save_model(model: keras.Model = None) -> None:
    '''
    Does the saving of the model. Depending on where the package is running at the moment.
    local: store on local disc
    gcp: store on Goggle Cloud Bucket
    mlflow: track and store with mlflow
    '''
    timestamp = time.strftime("%Y%m%d-%H%M%S")


    # Save model locally in any case:
    model_path = os.path.join(ROOT_PATH, 'models', f'{timestamp}.keras')
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET=='gcs':

        model_filename = model_path.split("/")[-1] # take only the part after the last /
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

    if MODEL_TARGET=='mlflow':
        pass
        #TODO: if needed


def load_model() -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    ((- or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only))

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(ROOT_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*") # lists all the existing models (all content) in the models folder

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # read model from Google Cloud Bucket
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated) # `.updated` gets the latest upload (not sorting alphabetically)
            latest_model_path_to_save = os.path.join(ROOT_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    elif MODEL_TARGET == "mlflow":
        pass
        #TODO: if needed, then add 'stage="Production"' to the functions params!


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{ROOT_PATH}/params/{current_timestamp}.pickle"
    "{ROOT_PATH}/metrics/{current_timestamp}.pickle"
    ___later___: if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    #if MODEL_TARGET == "mlflow":
    #    if params is not None:
    #        mlflow.log_params(params)
    #    if metrics is not None:
    #        mlflow.log_metrics(metrics)
    #    print("✅ Results saved on MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(ROOT_PATH, 'models', "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(ROOT_PATH, 'models', "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")
