FROM python:3.9.16

RUN apt-get update && apt-get install gunicorn -y

WORKDIR /app
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
COPY . .

CMD [ "gunicorn", "-k" , "eventlet", "-b", "0.0.0.0:5000", "-w", "1", "-t", "300", "backend.app_live_whisper:app"]
