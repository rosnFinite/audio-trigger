import json
import os

import nidaqmx
import socketio
import time
from threading import Thread, Event
from socketio.exceptions import ConnectionError

import logging

import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import collections
import plotly.graph_objs as go
from typing import List, Optional, Tuple, Union

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
        self.stop_event = Event()

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
            return
        logging.info("Starting audio stream thread...")
        self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stop_event = Event()
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
                 freq_bounds: Tuple[float, float] = (150.0, 1700.0),
                 dba_bin_size: int = 5,
                 dba_bounds: Tuple[int, int] = (35, 115),
                 buffer_size: float = 1.0,
                 channels: int = 1,
                 rate: int = 44100,
                 chunksize: int = 1024,
                 socket=None):
        super().__init__(buffer_size, rate, channels, chunksize)
        self.calib_factors = self.__load_calib_factors(dba_calib_file) if dba_calib_file is not None else None
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
        self.grid = Grid(semitone_bin_size, freq_bounds, dba_bin_size, dba_bounds, min_q_score, self.socket)
        self.instance_settings = {
            "sampleRate": rate,
            "bufferSize": buffer_size,
            "chunkSize": chunksize,
            "channels": channels,
            "qualityScore": min_q_score,
            "freqBounds": freq_bounds,
            "semitoneBinSize": semitone_bin_size,
            "dbaBounds": dba_bounds,
            "dbaBinSize": dba_bin_size
        }
        logging.info(f"Successfully created trigger: {self.instance_settings}")
        # check for ni daq board
        """
        self.daq = None
        if len(nidaqmx.system.System.local().devices) == 0:
            logging.info("No connected DAQ-Board found. Continue without...")
        else:
            logging.info(f"DAQ-Board {nidaqmx.system.System.local().devices[0]} connected.")
            self.daq = nidaqmx.system.System.local().devices[0]
        """

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
            return
        logging.info("Starting audio stream thread...")
        self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stop_event = Event()
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
    def __init__(self,
                 semitone_bin_size: int,
                 freq_bounds: Tuple[float, float],
                 dba_bin_size: int,
                 dba_bounds: Tuple[int, int],
                 min_q_score: float,
                 socket=None):
        self.__last_data_tuple: Optional[Tuple[int, int]] = None
        self.freq_bins_lb: List[float] = self.__calc_freq_lower_bounds(semitone_bin_size, freq_bounds)
        self.dba_bins_lb: List[int] = self.__calc_dba_lower_bounds(dba_bin_size, dba_bounds)
        logging.info(f"Created voice field with {len(self.freq_bins_lb)}[frequency bins] x {len(self.dba_bins_lb)}[dba bins].")
        self.min_q_score: float = min_q_score
        self.grid: List[List[Optional[float]]] = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        self.socket = socket

    def __is_bounds_valid(self, bounds: Union[Tuple[float, float],Tuple[int, int]]):
        if len(bounds) != 2:
            return False
        if bounds[0] == bounds[1]:
            return False
        return True

    def __calc_freq_lower_bounds(self, semitone_bin_size: int, freq_bounds: Tuple[float, float]) -> List[float]:
        if not self.__is_bounds_valid(freq_bounds):

            logging.critical(f"Provided frequency bounds are not valid. Tuple of two different values required. "
                             f"Got {freq_bounds}")
            raise ValueError("Provided frequency bounds are not valid.")
        # arbitrary start point for semitone calculations
        lower_bounds = [min(freq_bounds)]
        while lower_bounds[-1] < max(freq_bounds):
            lower_bounds.append(round(np.power(2, semitone_bin_size / 12) * lower_bounds[-1], 3))
        return lower_bounds

    def __calc_dba_lower_bounds(self, dba_bin_size: int, dba_bounds: Tuple[int, int]) -> List[int]:
        if not self.__is_bounds_valid(dba_bounds):
            logging.critical(f"Provided db(A) bounds are not valid. Tuple of two different values required. "
                             f"Got {dba_bounds}")
            raise ValueError("Provided db(A) bounds are not valid.")
        lower_bounds = [min(dba_bounds)]
        while lower_bounds[-1] < max(dba_bounds):
            lower_bounds.append(lower_bounds[-1] + dba_bin_size)
        return lower_bounds

    def __create_socket_payload(self):
        return {str(idx): freqs for idx, freqs in enumerate(self.grid)}

    def reset_grid(self):
        logging.debug("GRID resetted")
        self.grid = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]

    def add_trigger(self, freq: float, dba: float, q_score: float) -> Optional[str]:
        # find corresponding freq and db bins
        freq_bin = np.searchsorted(self.freq_bins_lb, freq)
        dba_bin = np.searchsorted(self.dba_bins_lb, dba)
        if freq_bin == 0 or dba_bin == 0:
            # value is smaller than the lowest bound
            return None
        logging.info(f"Voice update - freq: {freq}[{freq_bin}], dba: {dba}[{dba_bin}], q_score: {q_score}")
        if self.socket is not None:
            self.socket.emit("voice", {
                "freq_bin": int(freq_bin-1),
                "dba_bin": int(dba_bin-1),
                "freq": float(freq),
                "dba": float(dba),
                "score": float(q_score)
            })
        self.__last_data_tuple = (freq_bin, dba_bin)
        if q_score > self.min_q_score:
            return None
        old_q_score = self.grid[dba_bin - 1][freq_bin - 1]
        if old_q_score is None:
            self.grid[dba_bin - 1][freq_bin - 1] = q_score
            logging.info(f"+ Grid entry added - q_score: {q_score}")
        else:
            if old_q_score > q_score:
                self.grid[dba_bin - 1][freq_bin - 1] = q_score
                logging.info(f"++ Grid entry updated - q_score: {old_q_score} -> {q_score}")
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
    trigger = Trigger(channels=1, buffer_size=0.2, dba_calib_file="./calibration/Behringer.json")
    trigger.start_trigger(1)

