import json
import os
import threading
import time
from threading import Thread

import socketio
import logging

import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import collections
import plotly.graph_objs as go
from typing import List, Optional, Tuple

from webapp.processing.fourier import get_dominant_freq, calc_quality_score, get_dba_level, fft

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class AudioRecorder:
    def __init__(self, buffer_size=10., rate=16000, channels=2, chunksize=1024):
        # "CHUNK" is the (arbitrarily chosen) number of frames the (potentially very long)
        self.chunksize = chunksize
        # channels == 1, only audio input | channels == 2, audio and egg input
        self.channels = channels
        # "RATE" is the "sampling rate", i.e. the number of frames per second
        self.rate = rate
        self.buffer_size = buffer_size
        self.frames = collections.deque([] * int((buffer_size * rate) / chunksize),
                                        maxlen=int((buffer_size * rate) / chunksize))
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording_devices = self.__load_recording_devices()
        self.recording_device = None
        self.stream_thread = None
        self.stream_thread_is_running = False
        self.stop_event = threading.Event()

    def get_audio_data(self):
        if self.channels == 1:
            return np.hstack(self.frames)
        return np.hstack(self.frames)[0::2]

    def get_egg_data(self):
        if self.channels != 2:
            return None
        return np.hstack(self.frames)[1::2]

    def __load_recording_devices(self):
        info = self.p.get_host_api_info_by_index(0)
        devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get('name'))
        logging.info("Recording devices loaded successfully.")
        return devices

    def __recording_callback(self, input_data, frame_count, time_info, flags):
        frame = np.frombuffer(input_data, dtype=np.int16)
        self.frames.append(frame)
        return input_data, pyaudio.paContinue

    def stream_process(self, input_device_index, cb):
        stream = self.p.open(format=pyaudio.paInt16,
                             channels=self.channels,
                             rate=self.rate,
                             input=True,
                             input_device_index=input_device_index,
                             frames_per_buffer=self.chunksize,
                             stream_callback=cb)
        while not self.stop_event.is_set():
            pass
        stream.close()

    def start_stream(self, input_device_index):
        if self.stream_thread_is_running:
            logging.info("Stream is already running.")
        logging.info("Starting audio stream thread...")
        self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stream_thread = Thread(target=self.stream_process, args=(input_device_index, self.__recording_callback))
        self.stream_thread.start()
        logging.info("Audio stream started successfully.")

    def stop_stream(self):
        if not self.stream_thread_is_running:
            logging.info("Stream is not currently running.")
            return
        logging.info("Stopping audio stream thread...")
        if self.stream_thread:
            self.stop_event.set()
            self.stream_thread.join()
        self.stream_thread_is_running = False
        self.recording_device = None
        logging.info("Audio stream stopped successfully.")

    def stop_stream_and_save_wav(self, save_path):
        self.stop_stream()
        logging.info("Saving recorded audio buffer...")
        location = f"{save_path}/out.wav"
        wav.write(location, self.rate, self.get_audio_data())
        logging.info(f"Audio buffer saved as {location}")

    def terminate(self):
        self.p.terminate()


class Trigger(AudioRecorder):
    def __init__(self,
                 rec_destination: str = time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                 dba_calib_file: Optional[str] = None,
                 min_q_score: float = 50,
                 semitone_bin_size: int = 2,
                 dba_bin_size: int = 5,
                 buffer_size: float = 1.0,
                 channels: int = 1,
                 rate: int = 44100,
                 chunksize: int = 1024,
                 socket=None):
        super().__init__(buffer_size, rate, channels, chunksize)
        self.calib_factors = self.__load_calib_factors(dba_calib_file) if dba_calib_file is not None else None
        self.grid = Grid(semitone_bin_size, dba_bin_size, min_q_score, socket)
        self.__rec_destination = f"{os.path.dirname(os.path.abspath(__file__))}/{rec_destination}"
        # check if trigger destination folder exists, else create
        self.__check_rec_destination()
        # create a websocket connection
        self.socket = socket
        if self.socket is not None:
            if self.socket.connected:
                logging.info("Websocket connected to trigger.")
            else:
                logging.info("SocketIO client without connection to server provided. Try to connect to default url "
                             "'http://localhost:5001'...")
                try:
                    self.socket.connect("http://localhost:5001")
                    logging.info("Websocket connection to default server successfully established.")
                except Exception:
                    logging.info("Websocket connection to default server failed.")

    def __check_rec_destination(self):
        if os.path.exists(self.__rec_destination):
            return
        os.makedirs(self.__rec_destination)

    def __load_calib_factors(self, dba_calib_file: str) -> dict:
        logging.info("Loading calibration factors...")
        with open(dba_calib_file) as f:
            corr_factors = json.load(f)
        logging.info("Calibration factors successfully loaded.")
        return {value[1]: value[2] for value in list(corr_factors.values())}

    def start_trigger(self, input_device_index: int):
        if self.stream_thread_is_running:
            logging.info("Stream is already running.")
        logging.info("Starting audio stream thread...")
        self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stream_thread = Thread(target=self.stream_process, args=(input_device_index, self.__trigger_callback))
        self.stream_thread.start()
        logging.info("Audio stream started successfully.")

    def __trigger_callback(self, input_data, frame_count, time_info, flags):
        # TODO: Check if emptying frames will lead to better results -> less overlap between trigger
        frame = np.frombuffer(input_data, dtype=np.int16)
        self.frames.append(frame)
        if len(self.frames) == self.frames.maxlen:
            data = self.get_audio_data()
            fourier, fourier_to_plot, abs_freq, w = fft(data, self.rate)
            dom_freq = get_dominant_freq(data, abs_freq=abs_freq, rate=self.rate, w=w)
            dba_level = get_dba_level(data, self.rate, corr_dict=self.calib_factors)
            q_score = calc_quality_score(abs_freq=abs_freq)
            filename = self.grid.add_trigger(dom_freq, dba_level, q_score)
            if filename is not None:
                wav.write(f"{self.__rec_destination}/{filename}", self.rate, data)
        return input_data, pyaudio.paContinue

    def stop_trigger(self):
        logging.info("Stopping trigger...")
        super().stop_stream()
        if self.socket is not None and self.socket.connected:
            self.socket.disconnect()
            logging.info("Websocket connection closed successfully.")
        logging.info("Trigger stopped successfully.")


class Grid:
    def __init__(self, semitone_bin_size: int, dba_bin_size: int, min_q_score: float, socket=None):
        self.__last_data_tuple: Optional[Tuple[int, int]] = None
        self.freq_bins_lb: List[float] = self.__calc_freq_lower_bounds(semitone_bin_size)
        self.dba_bins_lb: List[int] = self.__calc_dba_lower_bounds(dba_bin_size)
        logging.info(f"Created voice field with {len(self.freq_bins_lb)}[frequency bins] x {len(self.dba_bins_lb)}[dba bins].")
        self.min_q_score: float = min_q_score
        self.grid: List[List[Optional[float]]] = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        self.socket = socket

    def __calc_freq_lower_bounds(self, semitone_bin_size: int) -> List[float]:
        # arbitrary start point for semitone calculations
        lower_bounds = [55.0]
        while lower_bounds[-1] < 2093:
            lower_bounds.append(np.power(2, semitone_bin_size / 12) * lower_bounds[-1])
        return lower_bounds

    def __calc_dba_lower_bounds(self, dba_bin_size: int) -> List[int]:
        lower_bounds = [45]
        while lower_bounds[-1] < 110:
            lower_bounds.append(lower_bounds[-1] + 5)
        return lower_bounds

    def __create_socket_payload(self):
        return {str(idx): freqs for idx, freqs in enumerate(self.grid)}

    def add_trigger(self, freq: float, dba: float, q_score: float) -> Optional[str]:
        # find corresponding freq and db bins
        freq_bin = np.searchsorted(self.freq_bins_lb, freq)
        dba_bin = np.searchsorted(self.dba_bins_lb, dba)
        if freq_bin == 0 or dba_bin == 0:
            # value is smaller than the lowest bound
            return None
        if self.socket is not None:
            logging.info(f"Voice update - freq: {freq}[{freq_bin}], dba: {dba}[{dba_bin}], q_score: {q_score}")
            self.socket.emit("voice-update", {
                "freq_bin": int(freq_bin),
                "dba_bin": int(dba_bin),
                "freq": float(freq),
                "dba": float(dba),
                "q_score": float(q_score)
            })
        self.__last_data_tuple = (freq_bin, dba_bin)
        if q_score > self.min_q_score:
            return None
        old_q_score = self.grid[dba_bin - 1][freq_bin - 1]
        if old_q_score is None:
            self.grid[dba_bin - 1][freq_bin - 1] = q_score
            logging.info(f"+ Grid entry added - q_score: {q_score}")
            if self.socket is not None:
                self.socket.emit("grid-update", self.__create_socket_payload())
        else:
            if old_q_score > q_score:
                self.grid[dba_bin - 1][freq_bin - 1] = q_score
                logging.info(f"++ Grid entry updated - q_score: {old_q_score} -> {q_score}")
                if self.socket is not None:
                    self.socket.emit("grid-update", self.__create_socket_payload())
        # return filename for added trigger point
        return self.__build_file_name(freq_bin - 1, dba_bin - 1)

    def __build_file_name(self, freq_bin: int, dba_bin: int):
        return f"freqLB_{freq_bin}_dbaLB_{dba_bin}.wav"

    def show_grid(self) -> go.Figure:
        fig = go.Figure(data=go.Heatmap(
            z=self.grid,
            hoverongaps=False
        ))
        if self.__last_data_tuple is not None:
            freq, dba = self.__last_data_tuple
            fig.add_trace(
                go.Scatter(x=[freq - 1], y=[dba - 1])
            )
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.freq_bins_lb))),
                ticktext=["%.2f" % freq_bin for freq_bin in self.freq_bins_lb]
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(self.dba_bins_lb))),
                ticktext=self.dba_bins_lb
            )
        )
        return fig


if __name__ == "__main__":
    trigger = AudioRecorder(channels=1)
    trigger.start_stream(input_device_index=1)
