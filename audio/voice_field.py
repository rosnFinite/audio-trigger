import concurrent.futures
import json
import logging
import os
import shutil
import threading
import time
from typing import Tuple, List, Optional, Union

import numpy as np
import parselmouth
import scipy.io.wavfile as wav

from audio.processing.utility import measure_praat_stats
from audio.daq_interface import DAQ_Device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VoiceField:
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
    min_score : float
        The minimum score required for a trigger to be added. Score is between 1 (best) and 0 (worst).
    retrigger_percentage_improvement: float
        The minimum quality score improvement [percentage] for retriggering an already set recording.
    socket : Optional
        The socket object for emitting voice updates. Defaults to None.
    """

    def __init__(self,
                 rec_destination: str,
                 semitone_bin_size: int = 2,
                 freq_bounds: Tuple[float, float] = (55, 1600),
                 dba_bin_size: int = 5,
                 dba_bounds: Tuple[int, int] = (35, 115),
                 min_score: float = 0.7,
                 retrigger_percentage_improvement: float = 0.1,
                 socket=None):
        self.rec_destination = rec_destination
        self.min_score = min_score
        self.retrigger_percentage_improvement = retrigger_percentage_improvement
        self.socket = socket
        self.daq = DAQ_Device(num_samples=1000, sample_rate=100000, analog_input_channels=["ai0"],
                              digital_trig_channel="pfi5")
        self._file_lock = threading.Lock()
        self.id = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        # Creation of lower bounds for frequency and dba heatmap bins as well as cutoff (highest allowed value)
        self.freq_bins_lb: List[float] = self.__calc_freq_lower_bounds(semitone_bin_size, freq_bounds)
        self.freq_cutoff: float = round(np.power(2, semitone_bin_size / 12) * self.freq_bins_lb[-1], 3)
        self.dba_bins_lb: List[int] = self.__calc_dba_lower_bounds(dba_bin_size, dba_bounds)
        self.dba_cutoff: float = self.dba_bins_lb[-1] + dba_bin_size
        # Creation of the grid
        self.grid: List[List[Optional[float]]] = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        logger.info(
            f"Created voice field with {len(self.freq_bins_lb)}[frequency bins] x {len(self.dba_bins_lb)}[dba bins].")

    @staticmethod
    def __create_versioned_dir(path: str) -> str:
        """Creates a new directory with an incremented version number if the directory already exists.
        If the directory does not exist, the path is returned as is.

        Parameters
        ----------
        path : str
            The path to the directory to increment.

        Returns
        -------
        str
            The path to the newly created directory.
        """
        original_path = path
        version = 0
        while os.path.exists(path):
            version += 1
            path = os.path.join(os.path.dirname(original_path), f"{os.path.basename(original_path)}_{version}")
        logger.info(f"Creating new directory: {path}")
        os.makedirs(path)
        return path

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

    def __submit_threadpool_task(self, task, *args):
        """Create a thread pool for concurrent execution."""
        try:
            self.pool.submit(task, *args)
        except RuntimeError:
            logger.warning("RuntimeError: ThreadPoolExecutor already shutdown occurred. This is not a critical "
                           "error: Cause by stopping and restarting same trigger instance. Reinitializing "
                           "threadpool...")
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            self.pool.submit(task, *args)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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
            logger.critical(f"Provided frequency bounds are not valid. Tuple of two different values required. "
                            f"Got {freq_bounds}")
            raise ValueError("Provided frequency bounds are not valid.")
        # arbitrary start point for semitone calculations
        lower_bounds = []
        current_freq = min(freq_bounds)
        while current_freq < max(freq_bounds):
            lower_bounds.append(current_freq)
            current_freq = round(np.power(2, semitone_bin_size / 12) * lower_bounds[-1], 3)
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
            logger.critical(f"Provided db(A) bounds are not valid. Tuple of two different values required. "
                            f"Got {dba_bounds}")
            raise ValueError("Provided db(A) bounds are not valid.")
        lower_bounds = [min(dba_bounds)]
        while lower_bounds[-1] < max(dba_bounds):
            lower_bounds.append(lower_bounds[-1] + dba_bin_size)
        return lower_bounds

    def reset_grid(self) -> str:
        """Resets the grid to its initial state and creates a new recording directory. The new directory will use the
        name of the previous directory with an incremented number at the end.
        <prefix>_<date>_<time> -> <prefix>_<date>_<time>_<version> where version is the incremented number.

        Returns
        -------
        str
            The path to the newly created recording directory.
        """
        self.grid = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        logger.info(f"Voice field grid reset. New recording directory created: {self.rec_destination}")
        return self.__create_versioned_dir(self.rec_destination)

    def save_data(self, save_dir: str, trigger_data: dict, praat_stats: dict, freq_bin: int, freq: float, dba_bin: int,
                  id: int) -> None:
        """Saves the data to the rec_destination folder.

        Parameters
        ----------
        save_dir: str
            Path to parent dictionary in which to store data.
        trigger_data : dict
            Dictionary containing the data to save.
        praat_stats : dict
            Dictionary containing the praat statistics.
        freq_bin : int
            The frequency bin.
        freq : float
            The exact frequency.
        dba_bin : int
            The db(A) bin.
        """
        start_total = time.time()
        logger.info(f"Thread [{id}]: starting update")
        logger.info(f"Thread [{id}]: acquiring lock")
        with self._file_lock:
            try:
                start_save = time.time()
                file_path = f"{save_dir}/input_data.npy"
                with open(file_path, "wb") as f:
                    np.save(f, trigger_data["data"])
                with open(f"{save_dir}/meta.json", "w") as f:
                    json_object = json.dumps({
                        "frequency_bin": int(freq_bin),
                        "bin_frequency": self.freq_bins_lb[freq_bin],
                        "exact_freq": freq,
                        "dba_bin": int(dba_bin),
                        "dba": self.dba_bins_lb[dba_bin],
                        "score": self.grid[dba_bin][freq_bin],
                        **praat_stats
                    }, indent=4)
                    f.write(json_object)
                wav.write(f"{save_dir}/input_audio.wav", trigger_data["sampling_rate"], trigger_data["data"])
                logger.info(f"Thread [{id}]: data saved to {file_path}, runtime: {time.time() - start_save} seconds.")
            except Exception as e:
                logger.error(f"Thread [{id}]: error saving data: {e}")
            finally:
                logger.info(f"Thread [{id}]: releasing lock")
        logger.info(f"Thread [{id}]: finished update, runtime: {time.time() - start_total:.4f} seconds.")

    def check_trigger(self, sound: parselmouth.Sound, freq: float, dba: float, score: float,
                      trigger_data: dict) -> bool:
        """Adds a trigger point to the recorder. If the quality score is below the threshold, the trigger point will be
        added to the grid. If a socket is provided, the trigger point will be emitted to the server.

        Parameters
        ----------
        sound: parselmouth.Sound
            The sound object containing the audio data.
        freq : float
            The frequency of the trigger point.
        dba : float
            The db(A) level of the trigger point.
        score : float
            The quality score of the trigger point.
        trigger_data : dict
            Dictionary containing data coming from audio recoder instance allowing to save additional
            information on trigger.
        """
        start = time.time()
        # if freq and/or dba are out of bounds
        if freq > self.freq_cutoff or dba > self.dba_cutoff:
            return False
        if freq < self.freq_bins_lb[0] or dba < self.dba_bins_lb[0]:
            return False

        # find corresponding freq and db bins e.g. [1,3,5,7], 4 -> 2, therefor -1 to reference the correct lower bound
        freq_bin = np.searchsorted(self.freq_bins_lb, freq) - 1
        dba_bin = np.searchsorted(self.dba_bins_lb, dba) - 1

        self.emit_voice(freq_bin, dba_bin, freq, dba, score)

        if score < self.min_score:
            return False

        existing_score = self.grid[dba_bin][freq_bin]
        # add trigger if no previous entry exists
        if existing_score is None:
            self.__add_trigger(sound, freq, freq_bin, dba_bin, score, trigger_data)
            logger.info(f"VOICE_FIELD entry added - score: {score}, "
                        f"runtime: {time.time() - start:.4f} seconds, save_data thread id: {self.id}.")
            return True
        # check if new score is [retrigger_percentage_improvement] % better than of existing score
        if existing_score < score and score / existing_score - 1 > self.retrigger_percentage_improvement:
            self.__add_trigger(sound, freq, freq_bin, dba_bin, score, trigger_data)
            logger.info(f"VOICE_FIELD entry updated - score: {existing_score} -> {score}, "
                        f"runtime: {time.time() - start:.4f} seconds, save_data thread id: {self.id}.")
            return True
        logger.info(f"Voice update - freq: {freq}[{freq_bin}], dba: {dba}[{dba_bin}], score: {score}, "
                    f"runtime: {time.time() - start:.6f} seconds")
        return False

    def __add_trigger(self, sound, freq, freq_bin, dba_bin, score, trigger_data):
        self.id += 1
        self.grid[dba_bin][freq_bin] = score
        data_dir = self.__create_versioned_dir(os.path.join(self.rec_destination, f"{dba_bin}_{freq_bin}"))

        # create a file named after newly added folder to parent dir of client recordings
        # this allows to easily find the latest added trigger for referencing corresponding camera images
        with open(os.path.join(os.path.split(self.rec_destination)[0], ".latest_trigger"), "w+") as f:
            f.write(data_dir)

        self.daq.start_acquisition(save_dir=data_dir)
        praat_stats = measure_praat_stats(sound, fmin=self.freq_bins_lb[0], fmax=self.freq_cutoff)
        self.emit_trigger(freq_bin, dba_bin, score, praat_stats)
        self.__submit_threadpool_task(self.save_data, data_dir, trigger_data, praat_stats, freq_bin, freq, dba_bin,
                                      self.id)

    def emit_voice(self, freq_bin: int, dba_bin: int, freq: float, dba: float, score: float) -> None:
        """Emit a voice update to the server.

        Parameters
        ----------
        freq : float
            The frequency of the voice update.
        dba : float
            The db(A) level of the voice update.
        score : float
            The quality score of the voice update.
        """
        if self.socket is None:
            return
        self.socket.emit("voice", {
            "freq_bin": int(freq_bin),
            "dba_bin": int(dba_bin),
            "freq": freq,
            "dba": dba,
            "score": score
        })

    def emit_trigger(self, freq_bin: int, dba_bin: int, score: float, praat_stats: dict) -> None:
        """Emit a trigger to the server.

        Parameters
        ----------
        freq_bin : int
            The frequency bin of the trigger.
        dba_bin : int
            The db(A) bin of the trigger.
        score : float
            The quality score of the trigger.
        praat_stats : dict
            The praat statistics of the trigger.
        """
        if self.socket is None:
            return
        self.socket.emit("trigger", {
            "freq_bin": int(freq_bin),
            "dba_bin": int(dba_bin),
            "score": float(score),
            "stats": {**praat_stats}
        })
