import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import Tuple, List, Optional, Union, Callable

import numpy as np
import parselmouth
import scipy.io.wavfile as wav

from src.audio.processing.utility import measure_praat_stats
from src.audio.daq_interface import DAQ_Device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VoiceField:
    """
    A class to represent and manage a voice field.

    This class provides methods to initialize the voice field, calculate frequency and dB(A) bin lower bounds,
    check indices, and update the field with new scores.

    Parameters
    ----------
    semitone_bin_size : int, optional
        The size of each semitone bin, by default 2.
    freq_bounds : Tuple[float, float], optional
        The lower and upper bounds of the frequency range, by default (55, 1600).
    db_bin_size : int, optional
        The size of each dB(A) bin, by default 5.
    db_bounds : Tuple[int, int], optional
        The lower and upper bounds of the dB(A) range, by default (35, 115).

    Attributes
    ----------
    freq_bins_lower_bounds : List[float]
        Lower bounds for the frequency bins.
    freq_cutoff : float
        The highest allowed frequency value.
    db_bins_lower_bounds : List[int]
        Lower bounds for the dB(A) bins.
    db_cutoff : int
        The highest allowed dB(A) value.
    field : List[List[Optional[float]]]
        The voice field matrix initialized with None values.

    Methods
    -------
    freq_min():
        Returns the minimum frequency of the voice field.
    freq_max():
        Returns the maximum frequency of the voice field.
    num_freq_bins():
        Returns the number of frequency bins in the voice field.
    db_min():
        Returns the minimum dB(A) level of the voice field.
    db_max():
        Returns the maximum dB(A) level of the voice field.
    num_db_bins():
        Returns the number of dB(A) bins in the voice field.
    __check_bounds(bounds: Union[Tuple[float, float], Tuple[int, int]]) -> bool:
        Checks if the provided bounds are valid.
    __check_indices_in_bounds(db_bin: int, freq_bin: int) -> bool:
        Checks if the given dB(A) and frequency bin indices are within bounds.
    __calc_freq_lower_bounds(semitone_bin_size: int, freq_bounds: Tuple[float, float]) -> List[float]:
        Calculates the lower bounds of the frequency bins.
    __calc_db_lower_bounds(db_bin_size: int, db_bounds: Tuple[int, int]) -> List[int]:
        Calculates the lower bounds of the dB(A) bins.
    reset_field() -> None:
        Resets the field to its initial state by setting all values back to None.
    get_field_score_at(db_bin: int, freq_bin: int) -> Optional[float]:
        Retrieves the score at the specified dB(A) and frequency bin.
    update_field_at(db_bin: int, freq_bin: int, score: float) -> None:
        Updates the field with a new score at the specified dB(A) and frequency bin.
    """

    def __init__(self,
                 semitone_bin_size: int = 2,
                 freq_bounds: Tuple[float, float] = (55, 1600),
                 db_bin_size: int = 5,
                 db_bounds: Tuple[int, int] = (35, 115)):
        # Creation of lower bounds for frequency and db heatmap bins as well as cutoff (highest allowed value)
        self.freq_bins_lower_bounds: List[float] = self.__calc_freq_lower_bounds(semitone_bin_size, freq_bounds)
        self.freq_cutoff: float = round(np.power(2, semitone_bin_size / 12) * self.freq_bins_lower_bounds[-1], 3)
        self.db_bins_lower_bounds: List[int] = self.__calc_db_lower_bounds(db_bin_size, db_bounds)
        self.db_cutoff: int = self.db_bins_lower_bounds[-1] + db_bin_size
        # Creation of the field
        self.field: List[List[Optional[float]]] = [[None] * self.num_freq_bins for _ in range(self.num_db_bins)]
        logger.info(
            f"Created voice field with {self.num_freq_bins}[frequency bins] x {self.num_db_bins}[db bins].")

    @property
    def freq_min(self) -> float:
        """
        The minimum frequency of the voice field.

        This property returns the minimum frequency value allowed in the voice field, as defined by the first element of
        the `freq_bins_lower_bounds` list.

        Returns
        -------
        float
            The minimum frequency of the voice field.
        """
        return self.freq_bins_lower_bounds[0]

    @property
    def freq_max(self) -> float:
        """
        The maximum frequency of the voice field.

        This property returns the maximum frequency value allowed in the voice field, as defined by the `freq_cutoff`.

        Returns
        -------
        float
            The maximum frequency of the voice field.
        """
        return self.freq_cutoff

    @property
    def num_freq_bins(self) -> int:
        """
        The number of frequency bins in the voice field.

        This property returns the total number of frequency bins defined in the voice field, based on the length of the
        `freq_bins_lower_bounds` list.

        Returns
        -------
        int
            The number of frequency bins in the voice field.
        """
        return len(self.freq_bins_lower_bounds)

    @property
    def db_min(self) -> int:
        """
        The minimum db(A) level of the voice field.

        This property returns the minimum dB(A) level allowed in the voice field, as defined by the first element of
        the `db_bins_lower_bounds` list.

        Returns
        -------
        int
            The minimum dB(A) level of the voice field.
        """
        return self.db_bins_lower_bounds[0]

    @property
    def db_max(self) -> int:
        """
        The maximum db(A) level of the voice field.

        This property returns the maximum dB(A) level allowed in the voice field, as defined by the `db_cutoff`.

        Returns
        -------
        int
            The maximum dB(A) level of the voice field.
        """
        return self.db_cutoff

    @property
    def num_db_bins(self) -> int:
        """
        The number of frequency bins in the voice field.

        This property returns the total number of dB(A) bins defined in the voice field, based on the length of the
        `db_bins_lower_bounds` list.

        Returns
        -------
        int
            The number of dB(A) bins in the voice field.
        """
        return len(self.freq_bins_lower_bounds)

    @staticmethod
    def __check_bounds(bounds: Union[Tuple[float, float], Tuple[int, int]]) -> bool:
        """
        Checks if the provided bounds are valid.

        This method verifies that the provided bounds are a tuple of two elements and that the elements are not equal.
        The first element must be less than the second element (lower bound, upper bound).

        Parameters
        ----------
        bounds : Union[Tuple[float, float], Tuple[int, int]]
            The bounds to check. It can be a tuple of two floats or two integers.

        Returns
        -------
        bool
            True if the bounds are valid, False otherwise.

        Examples
        --------
        >>> __check_bounds((1.0, 2.0))
        True

        >>> __check_bounds((3, 3))
        False

        >>> __check_bounds((5,2))
        False
        """
        if len(bounds) != 2:
            return False
        if bounds[0] == bounds[1]:
            return False
        return True

    def __calc_freq_lower_bounds(self, semitone_bin_size: int, freq_bounds: Tuple[float, float]) -> List[float]:
        """
        Calculates the lower bounds of the frequency bins.

        This method calculates the lower bounds for frequency bins based on the provided semitone bin size and
        frequency range.

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

        Raises
        ------
        ValueError
            If the provided frequency bounds are not valid.

        Examples
        --------
        >>> __calc_freq_lower_bounds(1, (20.0, 20000.0))
        [20.0, 21.0, 22.3, 23.6, ..., 19999.0]
        """
        if not self.__check_bounds(freq_bounds):
            logger.critical(f"Provided frequency bounds are not valid. Tuple of two different values required. "
                            f"Got {freq_bounds}")
            raise ValueError(f"Provided frequency bounds {freq_bounds} are not valid.")
        # arbitrary start point for semitone calculations
        lower_bounds = []
        current_freq = min(freq_bounds)
        while current_freq < max(freq_bounds):
            lower_bounds.append(current_freq)
            current_freq = round(np.power(2, semitone_bin_size / 12) * lower_bounds[-1], 3)
        return lower_bounds

    def __calc_db_lower_bounds(self, db_bin_size: int, db_bounds: Tuple[int, int]) -> List[int]:
        """
        Calculates the lower bounds of the db(A) bins.

        This method calculates the lower bounds for db(A) bins based on the provided db(A) bin size and
        db(A) range.

        Parameters
        ----------
        db_bin_size : int
            The size of each db(A) bin.
        db_bounds : Tuple[int, int]
            The lower and upper bounds of the db(A) range.

        Returns
        -------
        List[int]
            The lower bounds of the db(A) bins.

        Raises
        ------
        ValueError
            If the provided db(A) bounds are not valid.

        Examples
        --------
        >>> __calc_db_lower_bounds(5, (20, 100))
        [20, 25, 30, 35, ..., 100]
        """
        if not self.__check_bounds(db_bounds):
            logger.critical(f"Provided db(A) bounds are not valid. Tuple of two different values required. "
                            f"Got {db_bounds}")
            raise ValueError(f"Provided db(A) bounds {db_bounds} are not valid.")
        lower_bounds = [min(db_bounds)]
        while lower_bounds[-1] < max(db_bounds):
            lower_bounds.append(lower_bounds[-1] + db_bin_size)
        return lower_bounds

    def reset_field(self) -> None:
        """
        Resets the field to its initial state by setting all values to back to None and keeping the shape.
        """
        self.field = [[None] * self.num_freq_bins for _ in range(self.num_db_bins)]

    def get_field_score_at(self, db_bin: int, freq_bin: int) -> Optional[float]:
        """
        Retrieves the score for the specified dB(A) and frequency bin.

        This method returns the score from the field at the given dB(A) and frequency bin indices if they are within
        bounds. If the indices are out of bounds, a LookupError is raised.

        Parameters
        ----------
        db_bin : int
            The dB(A) bin index to check.
        freq_bin : int
            The frequency bin index to check.

        Returns
        -------
        Optional[float]
            The score at the specified dB(A) and frequency bin, or None if there is no score.

        Raises
        ------
        LookupError
            If the provided dB(A) or frequency bin indices are out of bounds.

        Examples
        --------
        >>> get_field_score_at(2, 3)
        0.85

        >>> get_field_score_at(10, 5)
        Traceback (most recent call last):
            ...
        LookupError: Index out of bounds for field shape (num_db_bins, num_freq_bins): db_bin: 10, freq_bin: 5
        """
        if 0 <= db_bin < self.num_db_bins and 0 <= freq_bin < self.num_freq_bins:
            return self.field[db_bin][freq_bin]
        else:
            raise LookupError(f"Index out of bounds for field shape ({self.num_db_bins}, {self.num_freq_bins}): "
                              f"db_bin: {db_bin}, freq_bin: {freq_bin}")

    def update_field_at(self, db_bin: int, freq_bin: int, score: float) -> None:
        """
        Updates the field with a new score at the specified dB(A) and frequency bin.

        This method updates the score in the field at the given dB(A) and frequency bin indices if they are within
        bounds. If the indices are out of bounds, a LookupError is raised.

        Parameters
        ----------
        db_bin : int
            The dB(A) bin index to update.
        freq_bin : int
            The frequency bin index to update.
        score : float
            The quality score to update the field with.

        Raises
        ------
        LookupError
            If the provided dB(A) or frequency bin indices are out of bounds.

        Examples
        --------
        >>> update_field_at(2, 3, 0.7)

        >>> update_field_at(10, 5, 0.7)
        Traceback (most recent call last):
            ...
        LookupError: Index out of bounds for field shape (num_db_bins, num_freq_bins): db_bin: 10, freq_bin: 5
        """
        if 0 <= db_bin < self.num_db_bins and 0 <= freq_bin < self.num_freq_bins:
            self.field[db_bin][freq_bin] = score
        else:
            raise LookupError(f"Index out of bounds for field shape ({self.num_db_bins}, {self.num_freq_bins}): "
                              f"db_bin: {db_bin}, freq_bin: {freq_bin}")

    def get_bin_indices(self, db: float, freq: float) -> Tuple[int, int]:
        """
        Calculates the indices of the frequency and dB(A) bins for the given frequency and dB(A) values.

        This method calculates the indices of the frequency and dB(A) bins in the voice field for the given frequency
        and dB(A) values. The indices are calculated based on the lower bounds of the bins and the provided values.

        Parameters
        ----------
        db : float
            The dB(A) value for which to find the bin index.
        freq : float
            The frequency value for which to find the bin index.

        Returns
        -------
        Tuple[int, int]
            The frequency bin index and dB(A) bin index for the given frequency and dB(A) values.

        Examples
        --------
        >>> get_bin_indices(50, 1000)
        (3, 16)
        """
        freq_bin = np.searchsorted(self.freq_bins_lower_bounds, freq, side="right") - 1
        db_bin = np.searchsorted(self.db_bins_lower_bounds, db, side="right") - 1
        return db_bin, freq_bin

    def is_in_bounds(self, db: float, freq: float):
        """
        Checks if the given frequency and dB(A) values are within the bounds of the voice field.

        This method checks if the given frequency and dB(A) values are within the bounds of the voice field. The values
        are within bounds if they are greater than or equal to the minimum values and less than the maximum values.

        Parameters
        ----------
        db : float
            The dB(A) value to check.
        freq : float
            The frequency value to check.

        Returns
        -------
        bool
            True if the values are within bounds, False otherwise.

        Examples
        --------
        >>> is_in_bounds(50, 1000)
        True

        >>> is_in_bounds(20, 20000)
        False
        """
        if self.db_min <= db <= self.db_max and self.freq_min <= freq <= self.freq_max:
            return True
        return False


class Trigger:
    """
    Class for controlling the trigger process of incoming audio data.

    The trigger process is responsible for the acquisition of all data sources (audio, EGG, etc.), processing the data,
    and saving it to the disk.

    Parameters
    ----------
    voice_field : VoiceField2
        An instance of the VoiceField2 class.
    rec_destination : str
        The path to the directory where recordings will be saved.
    min_score : float, optional
        The minimum quality score required to add a trigger, by default 0.7.
    retrigger_percentage_improvement : float, optional
        The percentage improvement required to update an existing trigger, by default 0.1.
    socket : optional
        A socket object for emitting events, by default None.

    Attributes
    ----------
    rec_destination : str
        The path to the directory where recordings will be saved.
    min_score : float
        The minimum quality score required to add a trigger.
    retrigger_percentage_improvement : float
        The percentage improvement required to update an existing trigger.
    socket : optional
        A socket object for emitting events.
    daq : DAQ_Device
        An instance of the DAQ_Device class for data acquisition.
    voice_field : VoiceField2
        An instance of the VoiceField2 class.
    _file_lock : threading.Lock
        A lock for ensuring thread-safe file operations.
    id : int
        An identifier for the current thread/task.
    _pool : concurrent.futures.ThreadPoolExecutor
        A thread pool for concurrent execution of tasks.

    Methods
    -------
    __create_versioned_dir(path: str) -> str
        Creates a versioned directory if the provided directory already exists.
    __submit_threadpool_task(task: Callable, *args) -> None
        Submits a task to a thread pool for concurrent execution.
    __perform_trigger(sound: parselmouth.Sound, freq, freq_bin: int, db_bin: int, score: float, trigger_data: dict) -> None
        Adds a new trigger and processes associated data.
    trigger(sound: parselmouth.Sound, freq: float, db: float, score: float, trigger_data: dict) -> bool
        Adds a trigger point to the recorder if certain conditions are met.
    save_data(save_dir: str, trigger_data: dict, praat_stats: dict, freq_bin: int, freq: float, db_bin: int, id: int) -> None
        Save the data to the specified directory.
    reset_field() -> None
        Resets the voice field and creates a new recording directory.
    emit_voice(freq_bin: int, db_bin: int, freq: float, db: float, score: float) -> None
        Emits a voice update event to the server if the socket is available.
    emit_trigger(freq_bin: int, db_bin: int, score: float, praat_stats: dict) -> None
        Emits a trigger event to the websocket server if the socket is available.
    """

    def __init__(self,
                 voice_field: VoiceField,
                 rec_destination: str,
                 min_score: float = 0.7,
                 retrigger_percentage_improvement: float = 0.1,
                 socket=None):
        self.rec_destination = rec_destination
        self.min_score = min_score
        self.retrigger_percentage_improvement = retrigger_percentage_improvement
        self.socket = socket
        self.daq = DAQ_Device(from_config=True)
        self.voice_field = voice_field

        # attributes for the thread pool
        self._file_lock = threading.Lock()
        self.id = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def __create_versioned_dir(path: str) -> str:
        """
        Creates a versioned directory.

        This method checks if the provided directory exists and increments the version number if it does. It ensures
        that a new directory is created and returns its path.

        Process description:
        --------------------
        If the path does not exist:
            - Create the provided directory.
            - Return the path.
        If the path exists:
            - Check the version number of directories with the same name <path>_<version>.
            - Increment the version number of the highest version found by 1.
            - Create the new directory with the incremented version number.
            - Return the path of the newly created directory.

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
        while os.path.exists(path):
            head, tail = os.path.split(original_path)
            splitted_tail = tail.split("_")
            # TODO: find a better way to handle timestamped directories, currently only 2 digit (up to 99) version numbers are supported
            if len(splitted_tail) > 1 and len(splitted_tail[-1]) < 3:
                old_version = int(splitted_tail[-1])
                path = os.path.join(head, "_".join(splitted_tail[:-1]) + f"_{old_version + 1}")
            else:
                path = os.path.join(head, f"{tail}_1")
        logger.info(f"Creating new directory: {path}")
        os.makedirs(path)
        return path

    def __submit_threadpool_task(self, task: Callable, *args) -> None:
        """
        Submits a task to a thread pool for concurrent execution.

        This method attempts to submit a given task to a thread pool. If the thread pool has already been shut down,
        a RuntimeError is caught, a warning is logged, and a new thread pool is created before resubmitting the task.
        The method ensures that the thread pool is initialized with a maximum of 4 workers.

        Parameters
        ----------
        task : callable
            The task (function) to be executed concurrently.
        *args
            Variable length argument list to be passed to the task.

        Notes
        -----
        - If a RuntimeError occurs due to the ThreadPoolExecutor being already shut down, the thread pool is
          reinitialized with a maximum of 4 workers, and the task is resubmitted.
        - This method handles non-critical errors related to stopping and restarting the same trigger instance,
          ensuring the thread pool is ready for new tasks.
        """
        try:
            self.pool.submit(task, *args)
        except RuntimeError:
            logger.warning("RuntimeError: ThreadPoolExecutor already shutdown occurred. This is not a critical "
                           "error: Cause by stopping and restarting same trigger instance. Reinitializing "
                           "threadpool...")
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
            self.pool.submit(task, *args)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def __perform_trigger(self, sound: parselmouth.Sound, freq, freq_bin: int, db_bin: int, score: float,
                          trigger_data: dict, cb_start_t) -> None:
        """
        Adds a new trigger and processes associated data.

        This method increments the thread ID, updates the grid with the given score, creates a versioned directory
        for the data, and starts data acquisition. It also measures speech statistics, emits the trigger, and saves
        the data using a thread pool.

        Parameters
        ----------
        sound : array_like
            The sound data to be analyzed.
        freq : float
            The frequency associated with the trigger.
        freq_bin : int
            The frequency bin index in the grid.
        db_bin : int
            The dB(A) bin index in the grid.
        score : float
            The score to be recorded in the grid.
        trigger_data : dict
            Additional data related to the trigger to be saved.

        Notes
        -----
        - This method increments the internal ID counter for triggers.
        - It updates the voice_field `field` attribute at the specified `db_bin` and `freq_bin` with the given `score`.
        - A versioned directory is created within `rec_destination` to store the acquired data.
        - The path to the versioned directory is written to a `.latest_trigger` file in the parent directory of
          `rec_destination` to facilitate locating the latest trigger.
        - Data acquisition is started using the `daq` attribute, saving data in the versioned directory.
        - Speech statistics are measured using the `measure_praat_stats` function with frequency limits set by
          `freq_bins_lb` and `freq_cutoff` inside the voice_field instance.
        - The `emit_trigger` method is called to handle the trigger event with the frequency bin, dB(A) bin, score,
          and measured statistics.
        - The `save_data` method is submitted to a thread pool for concurrent execution, passing all relevant
          parameters and the newly incremented trigger ID.
        """
        self.id += 1
        self.voice_field.update_field_at(db_bin, freq_bin, score)

        daq_data = None
        try:
            logger.debug(f"RUNTIME RECORDING -> ACQUISITON {time.time() - cb_start_t:.4f}s")
            daq_data = self.daq.start_acquisition()
        except AttributeError as e:
            logger.critical(e)

        data_dir = self.__create_versioned_dir(os.path.join(self.rec_destination, f"{db_bin}_{freq_bin}"))
        # create a file named after newly added folder to parent dir of client recordings
        # this allows to easily find the latest added trigger for referencing corresponding camera images
        with open(os.path.join(os.path.split(self.rec_destination)[0], ".latest_trigger"), "w+") as f:
            f.write(data_dir)

        if daq_data is not None:
            self.daq.save_as_csv(daq_data, data_dir)

        praat_stats = measure_praat_stats(sound, fmin=self.voice_field.freq_min, fmax=self.voice_field.freq_max)
        self.emit_trigger(freq_bin, db_bin, score, praat_stats)
        self.__submit_threadpool_task(self.save_data, data_dir, trigger_data, praat_stats, freq_bin, freq, db_bin,
                                      self.id)

    def trigger(self, sound: parselmouth.Sound, freq: float, db: float, score: float,
                trigger_data: dict, cb_start_t) -> bool:
        """
        Adds a trigger point to the recorder if certain conditions are met.

        This method checks the provided frequency and dB(A) level against specified bounds. If they are within
        bounds and the quality score meets the threshold, the trigger point is added to the grid and potentially
        emitted to the server.

        Parameters
        ----------
        sound : parselmouth.Sound
            The sound object containing the audio data.
        freq : float
            The frequency of the trigger point.
        db : float
            The dB(A) level of the trigger point.
        score : float
            The quality score of the trigger point.
        trigger_data : dict
            Dictionary containing data from the audio recorder instance, allowing additional information on the
            trigger to be saved.

        Returns
        -------
        bool
            True if a new trigger point was added to the grid or an existing one was updated, False otherwise.

        Notes
        -----
        - The method starts by checking if the frequency and dB(A) level are within acceptable bounds.
        - If the score is below the minimum threshold (`self.min_score`), the method returns False.
        - The frequency and dB(A) bins are determined using the lower bound arrays (`self.voice_field.freq_bins_lb` and
          `self.voice_field.db_bins_lb`).
        - The method emits a voice event with the frequency bin, dB(A) bin, frequency, dB(A) level, and score.
        - If no existing score is present in the field at the determined bins, a new trigger is added.
        - If an existing score is present, the new score must be sufficiently better than the existing score
          (based on `self.retrigger_percentage_improvement`) to update the trigger.
        - Logging is performed at various stages to provide runtime information and debugging details.

        Examples
        --------
        >>> trigger(sound, 440.0, 70.0, 0.85, trigger_data)
        True
        """
        start = time.time()
        # check if freq and db are within bounds
        if not self.voice_field.is_in_bounds(db, freq):
            return False

        db_bin, freq_bin = self.voice_field.get_bin_indices(db, freq)

        self.emit_voice(freq_bin, db_bin, freq, db, score)

        if score < self.min_score:
            return False

        existing_score = self.voice_field.get_field_score_at(db_bin, freq_bin)
        # add trigger if no previous entry exists
        if existing_score is None:
            logger.debug(f"RUNTIME RECORDING -> __PERFORM_TRIGGER {time.time() - cb_start_t:.4f}s")
            self.__perform_trigger(sound, freq, freq_bin, db_bin, score, trigger_data, cb_start_t)
            logger.info(f"VOICE_FIELD entry added - score: {score}, "
                        f"runtime: {time.time() - start:.4f} seconds, save_data thread id: {self.id}.")
            return True
        # check if new score is [retrigger_percentage_improvement] % better than of existing score
        if existing_score < score and score / existing_score - 1 >= self.retrigger_percentage_improvement:
            self.__perform_trigger(sound, freq, freq_bin, db_bin, score, trigger_data)
            logger.info(f"VOICE_FIELD entry updated - score: {existing_score} -> {score}, "
                        f"runtime: {time.time() - start:.4f} seconds, save_data thread id: {self.id}.")
            return True
        logger.info(f"Voice update - freq: {freq}[{freq_bin}], db: {db}[{db_bin}], score: {score}, "
                    f"runtime: {time.time() - start:.6f} seconds")
        return False

    def save_data(self, save_dir: str, trigger_data: dict, praat_stats: dict, freq_bin: int, freq: float, db_bin: int,
                  id: int) -> None:
        """
        Save the data to the specified directory.

        This method saves the provided trigger data, praat statistics, and metadata to files within the given
        directory. It ensures thread-safe access using a lock and logs the process.

        Parameters
        ----------
        save_dir : str
            Path to the directory in which to store the data.
        trigger_data : dict
            Dictionary containing the data to save. Expected keys are "audio" and optionally "egg".
        praat_stats : dict
            Dictionary containing the praat statistics.
        freq_bin : int
            The frequency bin index.
        freq : float
            The exact frequency.
        db_bin : int
            The dB(A) bin index.
        id : int
            The identifier for the current thread/task.

        Notes
        -----
        - The method starts by logging the initiation of the data saving process and acquires a lock to ensure
          thread-safe file operations.
        - The `trigger_data["audio"]` and optionally `trigger_data["egg"]` arrays are saved as `.npy` files in the
          specified directory.
        - Metadata, including frequency bin, exact frequency, dB(A) bin, score, and praat statistics, is saved as
          a JSON file.
        - The audio data is also saved as a `.wav` file using the provided sampling rate.
        - The method logs the runtime for saving the data and any errors encountered during the process.

        Examples
        --------
        save_data("/path/to/save_dir", trigger_data, praat_stats, 3, 440.0, 2, 1)
        """
        start_total = time.time()
        logger.info(f"Thread [{id}]: starting update")
        logger.info(f"Thread [{id}]: acquiring lock")
        with self._file_lock:
            try:
                start_save = time.time()
                file_path = f"{save_dir}/audio.npy"
                with open(file_path, "wb") as f:
                    np.save(f, trigger_data["audio"])
                if trigger_data["egg"] is not None:
                    file_path = f"{save_dir}/egg.npy"
                    with open(file_path, "wb") as f:
                        np.save(f, trigger_data["egg"])
                with open(f"{save_dir}/meta.json", "w") as f:
                    json_object = json.dumps({
                        "frequency_bin": int(freq_bin),
                        "bin_frequency": self.voice_field.freq_bins_lower_bounds[freq_bin],
                        "exact_freq": freq,
                        "dba_bin": int(db_bin),
                        "dba": self.voice_field.db_bins_lower_bounds[db_bin],
                        "score": self.voice_field.get_field_score_at(db_bin, freq_bin),
                        **praat_stats
                    }, indent=4)
                    f.write(json_object)
                wav.write(f"{save_dir}/input_audio.wav", trigger_data["sampling_rate"], trigger_data["audio"])
                logger.info(f"Thread [{id}]: data saved to {file_path}, runtime: {time.time() - start_save} seconds.")
            except Exception as e:
                logger.error(f"Thread [{id}]: error saving data: {e}")
            finally:
                logger.info(f"Thread [{id}]: releasing lock")
        logger.info(f"Thread [{id}]: finished update, runtime: {time.time() - start_total:.4f} seconds.")

    def reset_field(self) -> str:
        """Resets the voice field and creates a new recording directory."""
        self.voice_field.reset_field()
        self.rec_destination = self.__create_versioned_dir(self.rec_destination)
        return self.rec_destination

    def emit_voice(self, freq_bin: int, db_bin: int, freq: float, db: float, score: float) -> None:
        """
        Emits a voice update event to the server if the socket is available otherwise does nothing.

        Parameters
        ----------
        freq_bin : float
            The corresponding frequency bin of the voice update.
        db_bin : float
            The corresponding db(A) bin of the voice update.
        freq : float
            The frequency of the voice update.
        db : float
            The db(A) level of the voice update.
        score : float
            The quality score of the voice update.
        """
        if self.socket is None:
            return
        self.socket.emit("voice", {
            "freq_bin": int(freq_bin),
            "dba_bin": int(db_bin),
            "freq": freq,
            "dba": db,
            "score": score
        })

    def emit_trigger(self, freq_bin: int, db_bin: int, score: float, praat_stats: dict) -> None:
        """
        Emits a trigger event to the websocket server if the socket is available otherwise does nothing.

        Parameters
        ----------
        freq_bin : int
            The frequency bin of the trigger.
        db_bin : int
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
            "dba_bin": int(db_bin),
            "score": float(score),
            "stats": {**praat_stats}
        })