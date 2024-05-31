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
from src.config_utils import CONFIG

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
    chunk_size : int
        The number of frames per buffer.


    Attributes
    ----------
    chunk_size : int
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

    def __init__(self, buffer_size: float = 10., rate: int = 16000, channels: int = 2, chunk_size: int = 1024):
        # "CHUNK" is the (arbitrarily chosen) number of frames the (potentially very long)
        self.chunk_size = chunk_size
        # channels == 1, only audio input | channels == 2, audio and egg input
        self.channels = channels
        # "RATE" is the "sampling rate", i.e. the number of frames per second
        self.rate = rate
        self.buffer_size = buffer_size
        self.frames = collections.deque([] * int((buffer_size * rate) / chunk_size),
                                        maxlen=int((buffer_size * rate) / chunk_size))
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
                             frames_per_buffer=self.chunk_size,
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

    def terminate(self) -> None:  # pragma: no cover
        """Terminate the PyAudio instance.
        """
        self.p.terminate()


class AudioTriggerRecorder(AudioRecorder):
    """A class for continuously recording audio data from a specified input device and triggering on specific audio events.

    Parameters
    ----------
    rec_destination : str
        The destination folder for saving the recorded audio files.
    dba_calib_file : Optional[str]
        The path to the file containing calibration factors for dB(A) level calculation.
    min_score : float
        The minimum quality score threshold for accepting a trigger.
    retrigger_percentage_improvement: float
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
    chunk_size : int
        The size of each audio chunk.
    trigger_timeout : float
        The timeout in seconds for after a trigger event.
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
                 retrigger_percentage_improvement: float = 0.1,
                 semitone_bin_size: int = 2,
                 freq_bounds: Tuple[float, float] = (150.0, 1700.0),
                 dba_bin_size: int = 5,
                 dba_bounds: Tuple[int, int] = (35, 115),
                 buffer_size: float = 1.0,
                 channels: int = 1,
                 rate: int = 44100,
                 chunk_size: int = 1024,
                 trigger_timeout: float = 1.0,
                 socket: Optional[socketio.Client] = None,
                 from_config: bool = False) -> None:
        # update default settings with config file settings if from_config is True
        if from_config:
            logger.info("Loading trigger settings from config file...")
            super().__init__(CONFIG["buffer_size"], CONFIG["rate"], CONFIG["channels"], CONFIG["chunk_size"])
            self.min_score = CONFIG["min_score"]
            self.rec_destination = CONFIG["client_recordings_path"]
            self.trigger_timeout = CONFIG["trigger_timeout"]
            retrigger_percentage_improvement = CONFIG["min_trigger_improvement"]
            semitone_bin_size = CONFIG["semitone_bin_size"]
            freq_bounds = (CONFIG["frequency_bounds"]["lower"], CONFIG["frequency_bounds"]["upper"])
            dba_bin_size = CONFIG["decibel_bin_size"]
            dba_bounds = (CONFIG["decibel_bounds"]["lower"], CONFIG["decibel_bounds"]["upper"])
        else:
            super().__init__(buffer_size, rate, channels, chunk_size)
            self.min_score = min_score
            self.rec_destination = os.path.join(os.path.dirname(os.path.abspath(__file__)), rec_destination)
            self.trigger_timeout = trigger_timeout

        # check if trigger destination folder exists, else create
        self.__check_rec_destination()

        self.calib_factors = self.__load_calib_factors(dba_calib_file) if dba_calib_file is not None else None

        self.__last_trigger_time = None

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
            dba_bounds=dba_bounds,
            min_score=self.min_score,
            retrigger_percentage_improvement=retrigger_percentage_improvement,
            socket=self.socket
        )
        self.init_settings = {
            "sampling_rate": self.rate,
            "save_location": self.rec_destination,
            "buffer_size": self.buffer_size,
            "chunk_size": self.chunk_size,
            "channels": self.channels,
            "min_score": self.min_score,
            "retrigger_percentage_improvement": retrigger_percentage_improvement,
            "freq_bounds": freq_bounds,
            "semitone_bin_size": semitone_bin_size,
            "dba_bounds": dba_bounds,
            "dba_bin_size": dba_bin_size
        }
        logger.info(f"Successfully created trigger: {self.init_settings}")

    @property
    def settings(self) -> dict:
        """Get the settings of the trigger instance.

        Returns
        -------
        dict
            The settings of the trigger instance.
        """
        return {**self.init_settings, "device": self.recording_device}

    def __check_rec_destination(self) -> None:
        """Check if the destination folder for the recorded audio files exists. If not, create it.
        """
        logger.info(f"Checking rec destination: {self.rec_destination}")
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
        if not dba_calib_file.endswith(".json"):
            raise ValueError("Invalid file format. JSON file required.")
        logger.info("Loading calibration factors...")
        with open(dba_calib_file) as f:
            corr_factors = json.load(f)
        logger.info("Calibration factors successfully loaded.")
        return {value[1]: value[2] for value in list(corr_factors.values())}

    def start_trigger(self, input_device_index: Optional[int] = None) -> None:  # pragma: no cover
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
        frame = np.frombuffer(input_data, dtype=np.int16)
        self.frames.append(frame)
        # disable trigger / voice processing after a trigger for 1 second
        # (time for the camera to be ready for next recording)
        if self.__last_trigger_time is not None:
            time_diff = time.time() - self.__last_trigger_time
            if time_diff < 1:
                logger.debug(f"AudioTriggerRecorder callback processing temporarily disabled. Time diff: {time_diff} < 1")
                return input_data, pyaudio.paContinue
            else:
                # update status to running if timeout is over
                if self.socket is not None:
                    self.socket.emit("status_update_complete", {"status": "running", "save_location": self.rec_destination})
                    self.__last_trigger_time = None

        if len(self.frames) == self.frames.maxlen:
            audio = self.get_audio_data()
            egg = self.get_egg_data()
            sound = parselmouth.Sound(audio, sampling_frequency=self.rate)
            score, dom_freq = calc_pitch_score(sound=sound,
                                               freq_floor=self.voice_field.freq_bins_lb[0],
                                               freq_ceiling=self.rate // 2)
            dba_level = get_dba_level(audio, self.rate, corr_dict=self.calib_factors)
            is_trig = self.voice_field.check_trigger(sound, dom_freq, dba_level, score,
                                                     trigger_data={"audio": audio, "egg": egg, "sampling_rate": self.rate})
            if is_trig:
                self.frames = collections.deque([] * int((self.buffer_size * self.rate) / self.chunk_size),
                                                maxlen=int((self.buffer_size * self.rate) / self.chunk_size))
                self.__last_trigger_time = time.time()
                if self.socket is not None:
                    # update status to waiting if trigger was detected
                    self.socket.emit("status_update_complete", {"status": "waiting", "save_location": self.rec_destination})
        return input_data, pyaudio.paContinue

    def stop_trigger(self) -> None:  # pragma: no cover
        """Stop the audio recording stream.
        """
        logger.info("Stopping trigger...")
        super().stop_stream()
        self.voice_field.pool.shutdown(wait=True)
        logger.info("AudioTriggerRecorder stopped successfully.")


if __name__ == "__main__":
    trigger = AudioTriggerRecorder(channels=1, buffer_size=0.2, dba_calib_file="../../calibration/Behringer.json", min_q_score=100)
    trigger.start_trigger(1)
