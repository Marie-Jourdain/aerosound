FROM python:3.10-slim

COPY aerosound aerosound
COPY models models
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

#prevent tensorflow from searching for a GPU
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=2

# local
CMD uvicorn aerosound.api.aero_api:app --host 0.0.0.0

# deploy
# CMD ["uvicorn", "aerosound.api.aero_api:app", "--host", "0.0.0.0", "--port", "8080"]
