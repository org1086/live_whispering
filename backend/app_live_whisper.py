
import ffmpeg
from typing import Union
import os
import base64
from webbrowser import get 
import whisper
from whisper import Whisper
import time
from datetime import datetime, timedelta
import webrtcvad
import numpy as np
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from queue import Queue
import pandas
import threading
from engineio.payload import Payload
import math

Payload.max_decode_packets = 50
UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
warnings.filterwarnings("ignore")

async_mode = 'threading'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)


vad = webrtcvad.Vad()
vad.set_mode(3)
input_queue = Queue()
output_queue = Queue()
frames = b''

# load whisper model
model = whisper.load_model("tiny")

def buildTranscribedDataResponse(transcribed_text: str, isPhraseComplete: bool, isFinal: bool):
    return {'data': transcribed_text, 'isPhraseComplete': isPhraseComplete, 'isFinal': isFinal}

def whisper_processing(model: Whisper, in_queue: Queue, socket: SocketIO):
        print("\ntranscribing from your buffers forever\n")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                socket.emit(
                    "speechData", 
                    buildTranscribedDataResponse('', True, True)
                )
                break
            
            if len(audio_frames) > 0:
                print (f"whisper_processing: audio_frames length={len(audio_frames)}")
                start = time.perf_counter()

                audio = np.frombuffer(audio_frames, np.int16).flatten().astype(np.float32) / 32768.0

                print (f"sample_length={len(audio) / 16000} in second.")
                audio = whisper.pad_or_trim(audio)
                print (f"sample_length_pad={ len(audio) / 16000} in second.")

                text = model.transcribe(audio)
                inference_time = time.perf_counter()-start

                print (f"text={text['text']}")
                if text['text'] != "":
                    # emit socket event to client with transcribed data
                    # data to emit: [text,sample_length,inference_time]
                    socket.emit(
                        "speechData",
                        buildTranscribedDataResponse(text['text'], True, False)
                    )
    
@socketio.on('binaryAudioData')
def stream(message):
    global frames
    global input_queue

    # message length = 2048 in bytes
    # print (f"message length={len(message['chunk'])}")

    if len(message["chunk"]) >= 480:
        values = [(message["chunk"])[i:i + 960] for i in range(0, len(message["chunk"]), 960)]

        is_speeches=[]
        for value in values[:-1]:
            is_speech = vad.is_speech(value, 16000, 480)
            is_speeches.append(is_speech)
        if any(is_speeches):
            frames += message["chunk"]
        else:
            if len(frames) >= 480:
                input_queue.put(frames)
                print (f"*********new frame input length={len(frames)}*******")
                frames = b'' 

@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    emit("connect", {"data": f"id: {request.sid} is connected"})       

@socketio.on('start')
def start():
    print("start")
    whisper_process = threading.Thread(target=whisper_processing, args=(
        model, input_queue, socketio))
    whisper_process.start()
    whisper_process.join()

@socketio.on('stop')
def stop():
    global input_queue
    print("stop")
    input_queue.put("close")

@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    emit("disconnect", f"user {request.sid} disconnected", broadcast=True)

# flask root endpoint
@app.route("/", methods=["GET"])
def welcome():
    return "Whispering something on air!"

if __name__ == "__main__":
    app.run(debug=True, port=5003, host="0.0.0.0")

