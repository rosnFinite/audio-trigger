import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import collections


class AudioRecorder:
    def __init__(self, buffer_size=10, rate=16000, chunksize=1024):
        # "CHUNK" is the (arbitrarily chosen) number of frames the (potentially very long)
        self.chunksize = chunksize
        # "RATE" is the "sampling rate", i.e. the number of frames per second
        self.rate = rate
        self.buffer_size = buffer_size
        self.frames = collections.deque([] * int((buffer_size*rate)/chunksize),
                                        maxlen=int((buffer_size*rate)/chunksize))
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording_devices = self.__load_recording_devices()

    def get_audio_data(self):
        return np.hstack(self.frames)

    def __load_recording_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get('name'))
        return devices

    def __recording_callback(self, input_data, frame_count, time_info, flags):
        self.frames.append(np.frombuffer(input_data, dtype=np.int16))
        return input_data, pyaudio.paContinue

    def start_stream(self, input_device_index):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  input_device_index=input_device_index,
                                  frames_per_buffer=self.chunksize,
                                  stream_callback=self.__recording_callback)
        if self.stream.is_active():
            print("Stream is active.")

    def stop_stream(self):
        if self.stream is None:
            raise RuntimeError("Stream has not been created.")
        if self.stream is not None and not self.stream.is_active():
            raise RuntimeError("Stream is currently not active.")
        self.stream.close()
        print("Stream stopped.")

    def stop_stream_and_save_wav(self, save_path):
        self.stop_stream()
        wav.write(f"{save_path}/out.wav", self.rate, self.get_audio_data())
        print(f"Stream stopped and file saved at '{save_path}/out.wav'")

    def terminate(self):
        self.p.terminate()
