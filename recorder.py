import logging
import os
import json
import time
from threading import Thread, Event
from typing import List, Optional, Tuple, Union, Callable

import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import collections
import plotly.graph_objs as go
import nidaqmx
import socketio
from socketio.exceptions import ConnectionError

from webapp.processing.fourier import get_dominant_freq, calc_quality_score, get_dba_level, fft

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class AudioRecorder:
    """A class for continuously recording audio data from a specified input device.

    Parameters
    ----------
    buffer_size : float
        The size of the audio buffer in seconds.
    rate : int
        The sampling rate in frames per second.
    channels : int
        The number of audio channels.
    chunksize : int
        The number of frames per buffer.


    Attributes
    ----------
    chunksize : int
        The number of frames per buffer.
    channels : int
        The number of audio channels.
    rate : int
        The sampling rate in frames per second.
    buffer_size : float
        The size of the audio buffer in seconds.
    frames : collections.deque
        A deque to store the recorded audio frames.
    p : pyaudio.PyAudio
        An instance of the PyAudio class.
    stream : pyaudio.Stream
        The audio stream for recording.
    recording_devices : List[str]
        A list of available recording devices.
    recording_device : int
        The index of the selected recording device.
    stream_thread : threading.Thread
        The thread for running the audio stream.
    stream_thread_is_running : bool
        Indicates whether the stream thread is running.
    stop_event : threading.Event
        An event to signal the stream thread to stop.
    """

    def __init__(self, buffer_size: float = 10., rate: int = 16000, channels: int = 2, chunksize: int = 1024):
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

    def get_audio_data(self) -> np.ndarray:
        """Get the recorded audio data.

        Returns
        -------
        numpy.ndarray
            The recorded audio data as a 1D array.
        """
        if self.channels == 1:
            return np.hstack(self.frames)
        return np.hstack(self.frames)[0::2]

    def get_egg_data(self) -> Optional[np.ndarray]:
        """Get the recorded egg data.

        Returns
        -------
        numpy.ndarray
            The recorded egg data as a 1D array, if two channels are recorded. Otherwise, returns None.
        """
        if self.channels != 2:
            return None
        return np.hstack(self.frames)[1::2]

    def __load_recording_devices(self) -> List[str]:
        """Load the available recording devices.

        Returns
        -------
        List[str]
            A list of available recording devices.
        """
        info = self.p.get_host_api_info_by_index(0)
        devices = []
        for i in range(0, info.get('deviceCount')):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                devices.append(self.p.get_device_info_by_host_api_device_index(0, i).get('name'))
        logging.info("Recording devices loaded successfully.")
        return devices

    def __recording_callback(self, input_data: bytes, frame_count: int, time_info: dict, flags: int) -> Tuple[bytes, int]:
        """The callback function for recording audio.

        Parameters
        ----------
        input_data : bytes
            The audio input data.
        frame_count : int
            The number of frames in the input data.
        time_info : dict
            The time information.
        flags : int
            The flags.

        Returns
        -------
        Tuple[bytes, int]
            A tuple containing the modified input data and the status code.

        """
        frame = np.frombuffer(input_data, dtype=np.int16)
        self.frames.append(frame)
        return input_data, pyaudio.paContinue

    def __stream_task(self, input_device_index: int, cb: Optional[Callable[[bytes, int, dict, int], Tuple[bytes, int]]] = None) -> None:
        """The task for running the audio recording stream. This function will run in a separate thread and will
        continuously record audio data from the specified input device.

        Parameters
        ----------
        input_device_index : int
            The index of the input device.
        cb : Optional[Callable]
            The callback function for recording audio.
        """
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

    def start_stream(self, input_device_index: int, cb: Optional[Callable[[bytes, int, dict, int], Tuple[bytes, int]]] = None) -> None:
        """Start the audio recording stream.

        Parameters
        ----------
        input_device_index : int
            The index of the input device.
        cb : Optional[Callable]
            The callback function for recording audio. If not provided, the default callback will be used.

        """
        if self.stream_thread_is_running:
            logging.info("Stream is already running.")
            return
        logging.info("Starting audio stream thread...")
        self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stop_event = Event()
        self.stream_thread = Thread(target=self.__stream_task,
                                    args=(input_device_index, self.__recording_callback if cb is None else cb))
        self.stream_thread.start()
        logging.info("Audio stream started successfully.")

    def stop_stream(self) -> None:
        """Stop the audio recording stream.
        """
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

    def stop_stream_and_save_wav(self, parent_dir: str) -> None:
        """Stop the audio recording stream and save the recorded audio buffer as a WAV file.
        The WAV file will be saved to the specified parent directory with the name "out.wav".

        Parameters
        ----------
        parent_dir : str
            The path to save the WAV file.
        """
        self.stop_stream()
        logging.info("Saving recorded audio buffer...")
        location = f"{parent_dir}/out.wav"
        wav.write(location, self.rate, self.get_audio_data())
        logging.info(f"Audio buffer saved as {location}")

    def terminate(self) -> None:
        """Terminate the PyAudio instance.
        """
        self.p.terminate()


class Trigger(AudioRecorder):
    """A class for continuously recording audio data from a specified input device and triggering on specific audio events.

    Parameters
    ----------
    rec_destination : str
        The destination folder for saving the recorded audio files.
    dba_calib_file : Optional[str]
        The path to the file containing calibration factors for dB(A) level calculation.
    min_q_score : float
        The minimum quality score threshold for accepting a trigger.
    semitone_bin_size : int
        The size of the semitone bins for frequency analysis.
    freq_bounds : Tuple[float, float]
        The lower and upper bounds of the frequency range for trigger detection.
    dba_bin_size : int
        The size of the dB(A) level bins for trigger detection.
    dba_bounds : Tuple[int, int]
        The lower and upper bounds of the dB(A) level range for trigger detection.
    buffer_size : float
        The buffer size in seconds for audio recording.
    channels : int
        The number of audio channels.
    rate : int
        The sample rate of the audio.
    chunksize : int
        The size of each audio chunk.
    socket : Optional
        The websocket connection object.

    Attributes
    ----------
    calib_factors : dict
        The calibration factors for dB(A) level calculation. None if no calibration file is provided.
    grid : Grid
        The grid object for trigger detection and visualization.
    instance_settings : dict
        The settings of the trigger instance.
    socket : socketio.Client
        The websocket connection object.
    """
    def __init__(self,
                 rec_destination: str = f"recordings/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
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
                 socket: Optional[socketio.Client] = None):
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

    def __check_rec_destination(self) -> None:
        """Check if the destination folder for the recorded audio files exists. If not, create it.
        """
        if os.path.exists(self.__rec_destination):
            return
        os.makedirs(self.__rec_destination)

    @staticmethod
    def __load_calib_factors(dba_calib_file: str) -> dict:
        """Load the calibration factors for dB(A) level calculation from a JSON file.

        Parameters
        ----------
        dba_calib_file : str
            The path to the file containing the calibration factors.
        """
        logging.info("Loading calibration factors...")
        with open(dba_calib_file) as f:
            corr_factors = json.load(f)
        logging.info("Calibration factors successfully loaded.")
        return {value[1]: value[2] for value in list(corr_factors.values())}

    def start_trigger(self, input_device_index: int) -> None:
        """Start the audio recording stream and trigger on specific audio events.
        """
        self.start_stream(input_device_index, self.__trigger_callback)

    def __trigger_callback(self, input_data: bytes, frame_count: int, time_info: dict, flags: int) -> Tuple[bytes, int]:
        """The callback function for recording audio and triggering on specific audio events.

        Parameters
        ----------
        input_data : bytes
            The audio input data.
        frame_count : int
            The number of frames in the input data.
        time_info : dict
            The time information.
        flags : int
            The flags.

        Returns
        -------
        Tuple[bytes, int]
            A tuple containing the modified input data and the status code.
        """
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

    def stop_trigger(self) -> None:
        """Stop the audio recording stream.
        """
        logging.info("Stopping trigger...")
        super().stop_stream()
        logging.info("Trigger stopped successfully.")


class Grid:
    """
    Represents a grid for storing and manipulating voice data.

    Parameters
    ----------
    semitone_bin_size : int
        The size of each semitone bin.
    freq_bounds : Tuple[float, float]
        The lower and upper bounds of the frequency range.
    dba_bin_size : int
        The size of each db(A) bin.
    dba_bounds : Tuple[int, int]
        The lower and upper bounds of the db(A) range.
    max_q_score : float
        The maximum quality score for a trigger to be added to the grid.
    socket : Optional
        The socket object for emitting voice updates. Defaults to None.
    """
    def __init__(self,
                 semitone_bin_size: int,
                 freq_bounds: Tuple[float, float],
                 dba_bin_size: int,
                 dba_bounds: Tuple[int, int],
                 max_q_score: float,
                 socket=None):
        self.__last_data_tuple: Optional[Tuple[int, int]] = None
        self.freq_bins_lb: List[float] = self.__calc_freq_lower_bounds(semitone_bin_size, freq_bounds)
        self.dba_bins_lb: List[int] = self.__calc_dba_lower_bounds(dba_bin_size, dba_bounds)
        logging.info(
            f"Created voice field with {len(self.freq_bins_lb)}[frequency bins] x {len(self.dba_bins_lb)}[dba bins].")
        self.min_q_score: float = max_q_score
        self.grid: List[List[Optional[float]]] = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        self.socket = socket
        # check for ni daq board
        self.daq = None
        try:
            if len(nidaqmx.system.System.local().devices) == 0:
                logging.info("No DAQ-Board connected. Continuing without...")
            else:
                logging.info(f"DAQ-Board {nidaqmx.system.System.local().devices[0]} connected.")
                self.daq = nidaqmx.system.System.local().devices[0]
        except nidaqmx.errors.DaqNotFoundError:
            logging.info("No NI-DAQmx installation found on this device. Continuing without...")

    @staticmethod
    def __is_bounds_valid(bounds: Union[Tuple[float, float], Tuple[int, int]]) -> bool:
        """Check if the provided bounds are valid.

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Tuple[int, int]]
            The bounds to check.

        Returns
        -------
        bool
            True if the bounds are valid, False otherwise.
        """
        if len(bounds) != 2:
            return False
        if bounds[0] == bounds[1]:
            return False
        return True

    @staticmethod
    def __build_file_name(freq_bin: int, dba_bin: int):
        """Build the filename for the recorded audio file."""
        return f"freqLB_{freq_bin}_dbaLB_{dba_bin}.wav"

    def __calc_freq_lower_bounds(self, semitone_bin_size: int, freq_bounds: Tuple[float, float]) -> List[float]:
        """Calculate the lower bounds of the frequency bins.

        Parameters
        ----------
        semitone_bin_size : int
            The size of each semitone bin.
        freq_bounds : Tuple[float, float]
            The lower and upper bounds of the frequency range.

        Returns
        -------
        List[float]
            The lower bounds of the frequency bins.
        """
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
        """Calculate the lower bounds of the db(A) bins.

        Parameters
        ----------
        dba_bin_size : int
            The size of each db(A) bin.
        dba_bounds : Tuple[int, int]
            The lower and upper bounds of the db(A) range.

        Returns
        -------
        List[int]
            The lower bounds of the db(A) bins.
        """
        if not self.__is_bounds_valid(dba_bounds):
            logging.critical(f"Provided db(A) bounds are not valid. Tuple of two different values required. "
                             f"Got {dba_bounds}")
            raise ValueError("Provided db(A) bounds are not valid.")
        lower_bounds = [min(dba_bounds)]
        while lower_bounds[-1] < max(dba_bounds):
            lower_bounds.append(lower_bounds[-1] + dba_bin_size)
        return lower_bounds
    
    def __set_daq_trigger(self):
        if self.daq is None:
            return
        with nidaqmx.Task(new_task_name="AudioTrigger") as trig_task:
            trig_task.do_channels.add_do_chan("/Dev1/PFI0")
            trig_task.write(True)
            trig_task.wait_until_done(timeout=1)
            trig_task.write(False)
            trig_task.stop()
            logging.info("DAQ trigger signal send successfully.")

    def __create_socket_payload(self) -> dict:
        """Create a payload for the socket event.

        Returns
        -------
        dict
            The payload for the socket event.
        """
        return {str(idx): freqs for idx, freqs in enumerate(self.grid)}

    def reset_grid(self) -> None:
        """Reset the grid to its initial state.
        """
        logging.debug("GRID resetted")
        self.grid = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]

    def add_trigger(self, freq: float, dba: float, q_score: float) -> Optional[str]:
        """Adds a trigger point to the recorder. If the quality score is below the threshold, the trigger point will be
        added to the grid. If a socket is provided, the trigger point will be emitted to the server.

        Parameters
        ----------
        freq : float
            The frequency of the trigger point.
        dba : float
            The db(A) level of the trigger point.
        q_score : float
            The quality score of the trigger point.

        Returns
        -------
        Optional[str]
            The filename of the recorded audio file if the trigger point was added to the grid. Otherwise, None.
        """
        # find corresponding freq and db bins
        freq_bin = np.searchsorted(self.freq_bins_lb, freq)
        dba_bin = np.searchsorted(self.dba_bins_lb, dba)
        if freq_bin == 0 or dba_bin == 0:
            # value is smaller than the lowest bound
            return None
        logging.info(f"Voice update - freq: {freq}[{freq_bin}], dba: {dba}[{dba_bin}], q_score: {q_score}")
        if self.socket is not None:
            self.socket.emit("voice", {
                "freq_bin": int(freq_bin - 1),
                "dba_bin": int(dba_bin - 1),
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
            self.__set_daq_trigger()
            logging.info(f"+ Grid entry added - q_score: {q_score}")
            if self.socket is not None:
                self.socket.emit("trigger", {
                    "freq_bin": int(freq_bin - 1),
                    "dba_bin": int(dba_bin - 1),
                    "score": float(q_score)
                })
        else:
            if old_q_score > q_score:
                self.grid[dba_bin - 1][freq_bin - 1] = q_score
                self.__set_daq_trigger()
                logging.info(f"++ Grid entry updated - q_score: {old_q_score} -> {q_score}")
                if self.socket is not None:
                    self.socket.emit("trigger", {
                        "freq_bin": int(freq_bin - 1),
                        "dba_bin": int(dba_bin - 1),
                        "score": float(q_score)
                    })
        # return filename for added trigger point
        return self.__build_file_name(freq_bin - 1, dba_bin - 1)

    def create_heatmap(self) -> go.Figure:
        """Generates a heatmap figure with grid data and adds a scatter trace if last data tuple is available.

        Returns
        -------
        go.Figure
            The generated heatmap figure.
        """
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
    trigger = Trigger(channels=1, buffer_size=0.2, dba_calib_file="./calibration/Behringer.json", min_q_score=100)
    trigger.start_trigger(1)
