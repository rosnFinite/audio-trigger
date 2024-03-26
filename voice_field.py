import concurrent.futures
import logging
import os
import shutil
import threading
from typing import Tuple, List, Optional, Union

import nidaqmx
import numpy as np


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
                 rec_destination: str,
                 socket=None):
        self.__rec_destination = rec_destination
        self.max_q_score: float = max_q_score
        self.socket = socket
        self.daq = self.__check_for_daq()
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
        logging.info(
            f"Created voice field with {len(self.freq_bins_lb)}[frequency bins] x {len(self.dba_bins_lb)}[dba bins].")

    @staticmethod
    def __check_for_daq() -> Optional[nidaqmx.system.Device]:
        """Check for a connected NI-DAQmx device.

        Returns
        -------
        Optional[nidaqmx.system.Device]
            The connected NI-DAQmx device or None if no device is connected.
        """
        try:
            if len(nidaqmx.system.System.local().devices) == 0:
                logging.info("No DAQ-Board connected. Continuing without...")
                return None
            else:
                logging.info(f"DAQ-Board {nidaqmx.system.System.local().devices[0]} connected.")
                return nidaqmx.system.System.local().devices[0]
        except nidaqmx.errors.DaqNotFoundError:
            logging.info("No NI-DAQmx installation found on this device. Continuing without...")
            return None

    @staticmethod
    def __build_file_name(freq: float, dba: int):
        """Build the filename for storing artifacts corresponding to a specific grid cell."""
        return f"{dba}_{int(np.round(freq, 2))}"

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

    def __set_daq_trigger(self) -> None:
        """Send a trigger signal to the DAQ device."""
        if self.daq is None:
            return
        with nidaqmx.Task(new_task_name="AudioTrigger") as trig_task:
            trig_task.do_channels.add_do_chan("/Dev1/PFI0")
            trig_task.write(True)
            trig_task.wait_until_done(timeout=1)
            trig_task.write(False)
            trig_task.stop()
            logging.info("DAQ trigger signal send successfully.")

    def reset_grid(self) -> None:
        """Reset the grid to its initial state and deletes corresponding recordings.
        """
        logging.debug("GRID resetted")
        self.grid = [[None] * len(self.freq_bins_lb) for _ in range(len(self.dba_bins_lb))]
        # removing stored recordings and creating new folder
        shutil.rmtree(self.__rec_destination)
        os.makedirs(self.__rec_destination)

    def save_data(self, plot_data: dict, freq_bin: int, dba_bin: int, id: int) -> None:
        """Saves the data to the rec_destination folder.

        Parameters
        ----------
        plot_data : dict
            Dictionary containing the data to save.
        freq_bin : int
            The frequency bin.
        dba_bin : int
            The db(A) bin.
        """
        logging.info(f"Thread [{id}]: starting update")
        folder_name = self.__build_file_name(self.freq_bins_lb[freq_bin - 1],
                                             self.dba_bins_lb[dba_bin - 1])
        logging.info(f"Thread [{id}]: acquiring lock")
        with self._file_lock:
            try:
                directory = f"{self.__rec_destination}/{folder_name}"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                file_path = f"{directory}/input_data.npy"
                with open(file_path, "wb") as f:
                    np.save(f, plot_data["data"])
                logging.info(f"Thread [{id}]: data saved to {file_path}")
            except Exception as e:
                logging.error(f"Thread [{id}]: error saving data: {e}")
            finally:
                logging.info(f"Thread [{id}]: releasing lock")
        logging.info(f"Thread [{id}]: finished update")

    def add_trigger(self, freq: float, dba: float, score: float, plot_data: dict) -> bool:
        """Adds a trigger point to the recorder. If the quality score is below the threshold, the trigger point will be
        added to the grid. If a socket is provided, the trigger point will be emitted to the server.

        Parameters
        ----------
        freq : float
            The frequency of the trigger point.
        dba : float
            The db(A) level of the trigger point.
        score : float
            The quality score of the trigger point.
        plot_data : dict
            Dictionary containing data needed for visualizations.
        """
        # if freq and/or dba are out of bounds
        if freq > self.freq_cutoff or dba > self.dba_cutoff:
            return False
        if freq < self.freq_bins_lb[0] or dba < self.dba_bins_lb[0]:
            return False

        # find corresponding freq and db bins e.g. [1,3,5,7], 4 -> 2, therefor -1 to reference the correct lower bound
        freq_bin = np.searchsorted(self.freq_bins_lb, freq)
        dba_bin = np.searchsorted(self.dba_bins_lb, dba)

        logging.info(f"Voice update - freq: {freq}[{freq_bin}], dba: {dba}[{dba_bin}], score: {score}")
        self.emit_voice(freq_bin, dba_bin, freq, dba, score)

        if score < 0:
            return False

        existing_score = self.grid[dba_bin - 1][freq_bin - 1]
        # add trigger if no previous entry exists
        if existing_score is None:
            self.id += 1
            self.grid[dba_bin - 1][freq_bin - 1] = score
            self.__set_daq_trigger()
            logging.info(f"+ {self.id} Grid entry added - score: {score}")
            self.emit_trigger(freq_bin, dba_bin, score)
            self.pool.submit(self.save_data, plot_data, freq_bin, dba_bin, self.id)
            return True
        # at least 10% better than previous entry
        if existing_score < score * 0.9:
            self.id += 1
            self.grid[dba_bin - 1][freq_bin - 1] = score
            self.__set_daq_trigger()
            logging.info(f"++ {self.id} Grid entry updated - score: {existing_score} -> {score}")
            self.emit_trigger(freq_bin, dba_bin, score)
            self.pool.submit(self.save_data, plot_data, freq_bin, dba_bin, self.id)
            return True
        return False

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
            "freq_bin": int(freq_bin - 1),
            "dba_bin": int(dba_bin - 1),
            "freq": freq,
            "dba": dba,
            "score": score
        })

    def emit_trigger(self, freq_bin: int, dba_bin: int, score: float) -> None:
        """Emit a trigger to the server.

        Parameters
        ----------
        freq_bin : int
            The frequency bin of the trigger.
        dba_bin : int
            The db(A) bin of the trigger.
        score : float
            The quality score of the trigger.
        """
        if self.socket is None:
            return
        self.socket.emit("trigger", {
            "freq_bin": int(freq_bin - 1),
            "dba_bin": int(dba_bin - 1),
            "score": float(score)
        })