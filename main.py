import time
import scipy.io.wavfile as wav
import pyaudio
import numpy
import plotly.graph_objs as go
from pynput import keyboard

RATE = 16000
CHUNKSIZE = 1024

frames = []  # A python-list of chunks(numpy.ndarray)


def recording_callback(input_data, frame_count, time_info, flags):
    frames.append(numpy.frombuffer(input_data, dtype=numpy.int16))
    return input_data, pyaudio.paContinue


# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE,
                stream_callback=recording_callback)


def on_press(key):
    if key == keyboard.Key.esc:
        stream.stop_stream()
        print(f"{key} pressed")


# keyboard listener to cancel recording
listener = keyboard.Listener(
    on_release=on_press)
listener.start()

while stream.is_active():
    time.sleep(0.1)

stream.close()
p.terminate()
data = numpy.hstack(frames)

wav.write('audio/out.wav', RATE, data)


fig = go.Figure(go.Scatter(x=[x for x in range(len(data))], y=data))
fig.show()