import os

MODEL_TARGET = os.environ.get('MODEL_TARGET')
DATA_FORMAT = os.environ.get('DATA_FORMAT') # np - numpy-array, csv - csv-file

#Google
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get('BUCKET_NAME')


# Docker
IMAGE = os.environ.get("IMAGE")
ARTIFACTSREPO = os.environ.get("ARTIFACTSREPO")
MEMORY = os.environ.get("MEMORY")

# Model Lifecycle
MLFLOW_TRACKING_URI=os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT=os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME=os.environ.get("MLFLOW_MODEL_NAME")

LOCAL_REGISTRY_PATH = '/home/jana/code/Marie-Jourdain/aerosound/models'
#LOCAL_REGISTRY_PATH = os.environ.get('LOCAL_REGISTRY_PATH')

#===============CONSTANTS====================
