import os
import time
from time import sleep
from datetime import datetime, timedelta
import warnings
import threading
import webrtcvad
import numpy as np
import whisper
from whisper import Whisper
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from queue import Queue
from engineio.payload import Payload
from tempfile import NamedTemporaryFile

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
#output_queue = Queue()
frames = b''

SAMPLE_RATE = 16000                     # hertz

# check if speech exists
MIN_CHUNK_SIZE = 480                    # in bytes
STEP_SIZE = 960                         # in bytes

# phrase complete and max buffer size (max_length=30s)
PHRASE_TIMEOUT = 3                      # in seconds
THIRTY_SECS_SIZE = 30*2*SAMPLE_RATE     # in bytes

# global variables
combined_bytes = bytes()                # in bytes
last_sample_timestamp = None            # timestamp
cache_sample = None                     # {'timestamp':...,'data':...}
isMove2NextChunk = False                # each chunk maximum 30s
isPhraseComplete = False                # boolean
lastCombinedTranscribedText = None      # string
isFinal = False                         # session finished fired from client

# load whisper model
model = whisper.load_model("tiny")
print("Model loaded.\n")

def buildTranscribedDataResponse():
    global isPhraseComplete
    global lastTranscribedText
    global isFinal

    return {
        'data': lastTranscribedText, 
        'isPhraseComplete': isPhraseComplete, 
        'isFinal': isFinal
    }

def popAtMost30SecondLengthSample() -> bytearray:
    '''
    Pop as much items from queue (in bytes) but less than 30 secs or phrase complete timeout.
    - Input queue is required.
    '''
    global combined_bytes
    global last_sample_timestamp
    global cache_sample
    global isMove2NextChunk
    global isPhraseComplete
    
    if isPhraseComplete:
        combined_bytes = bytes()
        last_sample_timestamp = None
        isPhraseComplete = False
    
    if isMove2NextChunk:
        combined_bytes = bytes()
        last_sample_timestamp = None
        isMove2NextChunk = False

    if cache_sample:
        last_sample_timestamp = cache_sample['timestamp']
        combined_bytes = cache_sample['data']
        cache_sample = None

    if not last_sample_timestamp:
        first_sample = input_queue.get()
        last_sample_timestamp = first_sample['timestamp']
        combined_bytes = first_sample['data']

    while (not input_queue.empty()) and (len(combined_bytes) < THIRTY_SECS_SIZE):
        current_sample = input_queue.get()
        current_sample_timestamp = current_sample['timestamp']
        current_sample_data = current_sample['data']
        
        if current_sample_timestamp - last_sample_timestamp < timedelta(seconds=PHRASE_TIMEOUT):
            if len(combined_bytes) + len(current_sample_data) <= THIRTY_SECS_SIZE:
                last_sample_timestamp = current_sample_timestamp
                combined_bytes += current_sample_data
            else:
                cache_sample = current_sample
                isMove2NextChunk = True
                break
        else:
            cache_sample = current_sample
            isPhraseComplete = True
            break
    
    return combined_bytes

def whisper_processing(model: Whisper, in_queue: Queue, socket: SocketIO):
        print("\nTranscribing from your buffers forever\n")
        while True:
            if in_queue.empty():
                sleep(0.05)
                continue

            audio_frames = popAtMost30SecondLengthSample()
            if audio_frames == "close":
                socket.emit(
                    "speechData", 
                    buildTranscribedDataResponse())
                break
            
            print (f"whisper_processing: audio_frames length={len(audio_frames)}")
            start = time.perf_counter()

            audio = np.frombuffer(audio_frames, np.int16).flatten().astype(np.float32) / 32768.0

            print (f"sample_length={len(audio) / SAMPLE_RATE} in second.")
            audio = whisper.pad_or_trim(audio)
            print (f"sample_length_pad={ len(audio) / SAMPLE_RATE} in second.")

            text = model.transcribe(audio)
            inference_time = time.perf_counter()-start

            print (f"text={text['text']}")
            if text['text'] != "":
                # emit socket event to client with transcribed data
                # data to emit: [text,sample_length,inference_time]
                socket.emit(
                    "speechData",
                    buildTranscribedDataResponse())                    

def isContainSpeech(message: bytearray) -> bool:
    values = [(message)[i:i + STEP_SIZE] 
                  for i in range(0, len(message), STEP_SIZE)]

    is_speeches=[]
    for value in values[:-1]:
        is_speech = vad.is_speech(value, SAMPLE_RATE, MIN_CHUNK_SIZE)
        is_speeches.append(is_speech)
    
    if any(is_speeches): return True
    else: return False

@socketio.on('binaryAudioData')
def stream(message):
    global frames
    global input_queue

    # message length = 2048 in bytes
    # print (f"message length={len(message['chunk'])}")

    if len(message["chunk"]) >= MIN_CHUNK_SIZE:
        if isContainSpeech(message["chunk"]):
            frames += message["chunk"]
        else:
            input_queue.put({'timestamp': datetime.utcnow(), 'data': frames})
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

