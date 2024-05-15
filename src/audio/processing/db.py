import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from src.audio.processing.utility import bisection, freezeargs


def get_dba_level(data: np.ndarray, rate: int, corr_dict: Optional[dict[str, float]] = None) -> float:
    """Calculates the average power output of the provided audio data via RMS (Root Mean Square). Currently, the A
    weighting is disabled due to high computational cost.

    Parameters
    ----------
    data : np.ndarray
        Numpy array of the recorded audio data.
    rate : int
        Sampling rate of the provided audio data.
    corr_dict : Optional[dict[str, float]]
        Dictionary containing db(A) value as keys and a list of corresponding microphone value and correction factor
        as values.

    Returns
    -------
    float
        The calculated power output of the provided audio data (RMS value).
    """
    # temporarily disable A weighting (high computational cost)
    # weighted_signal = A_weight(data, fs=rate)
    rms_value = np.sqrt(np.mean(np.power(np.abs(data).astype(np.int32), 2)))
    result = 20 * np.log10(rms_value)
    if corr_dict is None:
        return result
    xp, corr_interp = interp_correction(corr_dict)
    idx = bisection(xp, result)
    if idx == -1:
        idx = 0
    else:
        idx -= 1
    return result + corr_interp[idx]


@freezeargs
@lru_cache
def interp_correction(corr_dict: dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates correction factors represented as a dictionary of key-value pairs. Key represents value
    recorded by the microphone and corresponding value the correction value for dB(A) calculations.

    Parameters
    ----------
    corr_dict : dict[str, float]
        Dictionary (Key-Value pairs) of microphone value and corresponding correction value.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing interpolated correction factors for microphone values and corresponding correction values.
    """
    # sort dict
    corr_dict = dict(sorted(corr_dict.items()))
    key_list = [float(i) for i in list(corr_dict.keys())]
    val_list = list(corr_dict.values())
    xp = np.linspace(math.floor(min(key_list)), math.ceil(max(key_list)))
    corr_interp = np.interp(xp, key_list, val_list)
    return xp, corr_interp
