import whisper
import time

model = whisper.load_model("medium")
start_time = time.time()
print (f"start time: {start_time}")
result = model.transcribe("eu.wav")
end_time = time.time()
print (f"end time: {end_time}")
print(result["text"])
print (f"time of transcribing: {end_time - start_time}")