import time
from math import log10

from recorder import AudioRecorder

recorder = AudioRecorder(buffer_size=1, rate=44100)

info = recorder.p.get_host_api_info_by_index(0)
data = []
for i in range(0, info.get('deviceCount')):
    if (recorder.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print(recorder.p.get_device_info_by_host_api_device_index(0, i).get('name'))

recorder.start_stream(input_device_index=1)

while recorder.stream.is_active():
    db = 20 * log10(recorder.rms)
    print(f"RMS: {recorder.rms} DB: {db+117}")
    time.sleep(1)

