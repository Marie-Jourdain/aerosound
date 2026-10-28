FROM python:3.10-slim

COPY aerosound aerosound
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# local
#CMD uvicorn aerosound.api.aero_api:app --host 0.0.0.0

# deploy
CMD ["uvicorn", "aerosound.api.aero_api:app", "--host", "0.0.0.0", "--port", "8080"]
