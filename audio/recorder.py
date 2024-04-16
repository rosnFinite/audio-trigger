import logging
import os
import json
import time
from threading import Thread, Event
from typing import List, Optional, Tuple, Callable

import parselmouth
import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import collections
import socketio

from .voice_field import VoiceField
from .processing.db import get_dba_level
from .processing.scoring import calc_pitch_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        logger.info("Recording devices loaded successfully.")
        return devices

    def __recording_callback(self, input_data: bytes, frame_count: int, time_info: dict, flags: int) -> Tuple[
        bytes, int]:
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

    def __stream_task(self, input_device_index: int,
                      cb: Optional[Callable[[bytes, int, dict, int], Tuple[bytes, int]]] = None) -> None:
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

    def start_stream(self,
                     input_device_index: Optional[int] = None,
                     cb: Optional[Callable[[bytes, int, dict, int], Tuple[bytes, int]]] = None) -> None:
        """Start the audio recording stream.

        Parameters
        ----------
        input_device_index : Optional[int]
            The index of the input device.
        cb : Optional[Callable]
            The callback function for recording audio. If not provided, the default callback will be used.

        """
        if self.stream_thread_is_running:
            logger.info("Stream is already running.")
            return
        logger.info("Starting audio stream thread...")
        if input_device_index is None and self.recording_device is None:
            raise ValueError("No input device index provided.")
        if input_device_index is not None:
            self.recording_device = input_device_index

        self.stream_thread_is_running = True
        self.stop_event = Event()
        self.stream_thread = Thread(target=self.__stream_task,
                                    args=(self.recording_device, self.__recording_callback if cb is None else cb))
        self.stream_thread.start()
        logger.info("Audio stream started successfully.")

    def stop_stream(self) -> None:
        """Stop the audio recording stream.
        """
        if not self.stream_thread_is_running:
            logger.info("Stream is not currently running.")
            return
        logger.info("Stopping audio stream thread...")
        if self.stream_thread:
            self.stop_event.set()
            self.stream_thread.join()
        self.stream_thread_is_running = False
        logger.info("Audio stream stopped successfully.")

    def stop_stream_and_save_wav(self, parent_dir: str) -> None:
        """Stop the audio recording stream and save the recorded audio buffer as a WAV file.
        The WAV file will be saved to the specified parent directory with the name "out.wav".

        Parameters
        ----------
        parent_dir : str
            The path to save the WAV file.
        """
        self.stop_stream()
        logger.info("Saving recorded audio buffer...")
        location = f"{parent_dir}/out.wav"
        wav.write(location, self.rate, self.get_audio_data())
        logger.info(f"Audio buffer saved to {location}")

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
    min_score : float
        The minimum quality score threshold for accepting a trigger.
    retrigger_score_threshold: float
        The minimum quality score improvement [percentage] for retriggering an already set recording.
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
    voice_field : VoiceField
        The grid object for trigger detection and visualization.
    instance_settings : dict
        The settings of the trigger instance.
    socket : socketio.Client
        The websocket connection object.
    """

    def __init__(self,
                 rec_destination: str = f"recordings/{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}",
                 dba_calib_file: Optional[str] = None,
                 min_score: float = 0.7,
                 retrigger_score_threshold: float = 0.1,
                 semitone_bin_size: int = 2,
                 freq_bounds: Tuple[float, float] = (150.0, 1700.0),
                 dba_bin_size: int = 5,
                 dba_bounds: Tuple[int, int] = (35, 115),
                 buffer_size: float = 1.0,
                 channels: int = 1,
                 rate: int = 44100,
                 chunksize: int = 1024,
                 socket: Optional[socketio.Client] = None):
        logger.info(f"Number of channels: {channels}")
        super().__init__(buffer_size, rate, channels, chunksize)
        self.min_score = min_score
        self.calib_factors = self.__load_calib_factors(dba_calib_file) if dba_calib_file is not None else None
        self.rec_destination = os.path.join(os.path.dirname(os.path.abspath(__file__)), rec_destination)
        # check if trigger destination folder exists, else create
        self.__check_rec_destination()
        # create a websocket connection
        self.socket = socket
        if self.socket is not None:
            if self.socket.connected:
                logger.info("Websocket connection established.")
            else:
                logger.warning("SocketIO client without connection to server provided. Try to connect to default url "
                               "'http://localhost:5001'...")
                try:
                    self.socket.connect("http://localhost:5001")
                    logger.info("Websocket connection to default server successfully established.")
                except Exception:
                    logger.critical("Websocket connection to default server failed.")
        self.voice_field = VoiceField(
            rec_destination=self.rec_destination,
            semitone_bin_size=semitone_bin_size,
            freq_bounds=freq_bounds,
            dba_bin_size=dba_bin_size,
            dba_bounds=dba_bounds,
            min_score=min_score,
            retrigger_score_threshold=retrigger_score_threshold,
            socket=self.socket
        )
        self.instance_settings = {
            "sampleRate": rate,
            "bufferSize": buffer_size,
            "chunkSize": chunksize,
            "channels": channels,
            "minScore": min_score,
            "reriggerScoreThreshold": retrigger_score_threshold,
            "freqBounds": freq_bounds,
            "semitoneBinSize": semitone_bin_size,
            "dbaBounds": dba_bounds,
            "dbaBinSize": dba_bin_size
        }
        logger.info(f"Successfully created trigger: {self.instance_settings}")
        self.debug_num_runs = 0
        self.debug_inner_runs = 0

    def __check_rec_destination(self) -> None:
        """Check if the destination folder for the recorded audio files exists. If not, create it.
        """
        print(f"Checking rec destination: {self.rec_destination}")
        if os.path.exists(self.rec_destination):
            return
        os.makedirs(self.rec_destination)

    @staticmethod
    def __load_calib_factors(dba_calib_file: str) -> dict:
        """Load the calibration factors for dB(A) level calculation from a JSON file.

        Parameters
        ----------
        dba_calib_file : str
            The path to the file containing the calibration factors.
        """
        logger.info("Loading calibration factors...")
        with open(dba_calib_file) as f:
            corr_factors = json.load(f)
        logger.info("Calibration factors successfully loaded.")
        return {value[1]: value[2] for value in list(corr_factors.values())}

    def start_trigger(self, input_device_index: Optional[int] = None) -> None:
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
            sound = parselmouth.Sound(data, sampling_frequency=self.rate)
            score, dom_freq = calc_pitch_score(sound=sound)
            dba_level = get_dba_level(data, self.rate, corr_dict=self.calib_factors)
            is_trig = self.voice_field.check_trigger(sound, dom_freq, dba_level, score,
                                                     trigger_data={"data": data, "sampling_rate": self.rate})
            if is_trig:
                self.frames = collections.deque([] * int((self.buffer_size * self.rate) / self.chunksize),
                                                maxlen=int((self.buffer_size * self.rate) / self.chunksize))
        return input_data, pyaudio.paContinue

    def stop_trigger(self) -> None:
        """Stop the audio recording stream.
        """
        logger.info("Stopping trigger...")
        super().stop_stream()
        self.voice_field.pool.shutdown(wait=True)
        logger.info("Trigger stopped successfully.")


if __name__ == "__main__":
    trigger = Trigger(channels=1, buffer_size=0.2, dba_calib_file="../calibration/Behringer.json", min_q_score=100)
    trigger.start_trigger(1)