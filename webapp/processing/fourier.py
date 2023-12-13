import librosa
import plotly.graph_objs as go
import numpy as np
import scipy
import math
from typing import Optional, Tuple
from functools import lru_cache

from webapp.processing.weighting import A_weight
from webapp.utility import bisection, freezeargs


def fft(data: np.array, rate: int):
    """Performs FFT on provided data.

    :param data: Numpy array of the recorded audio data.
    :param rate: Sampling rate of the recorder.
    :return:
    """
    fourier = scipy.fft.fft(data)
    fourier_to_plot = fourier[0:len(fourier) // 2]
    abs_freq = np.abs(fourier_to_plot)
    w = np.linspace(0, rate, len(fourier))[0:len(fourier) // 2]
    return fourier, fourier_to_plot, abs_freq, w


def get_note(data: Optional[np.ndarray] = None,
             rate: Optional[int] = None,
             abs_freq: Optional[np.ndarray] = None,
             w: Optional[np.ndarray] = None) -> str:
    """Returns the strongest represented musical note.

    :param w:
    :param abs_freq:
    :param data: Numpy array of the recorded audio data.
    :param rate: Sampling rate of the recorder.
    :return:
    """
    if abs_freq is None or w is None:
        if data is not None and rate is not None:
            _, _, abs_freq, w = fft(data, rate)
        else:
            raise ValueError("No fft_data was provided. Missing data and rate. ")
    amp = abs_freq.argmax()
    freq = w[amp]
    note = librosa.hz_to_note(freq)
    return note


def plot_abs_fft(data: np.ndarray, rate: int) -> Tuple[go.Figure, str, float]:
    """Performs FFT on provided data and returns a go.Figure visualizing the frequency domain.

    :param data: Numpy array of the recorded audio data.
    :param rate: Sampling rate of the recorder.
    :return:
    """
    fourier, fourier_to_plot, abs_freq, w = fft(data, rate)
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    abs_freq = np.abs(fourier_to_plot)
    freq_fig.add_trace(go.Scatter(x=w, y=abs_freq))
    note = get_note(abs_freq=abs_freq, w=w)
    score = calc_quality_score(abs_freq=abs_freq)
    return freq_fig, note, score


def calc_quality_score(data: Optional[np.ndarray] = None,
                       rate: Optional[int] = None,
                       abs_freq: Optional[np.ndarray] = None) -> float:
    """Calculates quality score denoting the strength of strongest frequency compared to other frequencies.
    Values closer to 0 denote higher quality/strength of the mainly detected frequency.

    :param abs_freq:
    :param data: Numpy array of the recorded audio data.
    :param rate: Sampling rate of the recorder.
    """
    if abs_freq is None:
        if data is not None and rate is not None:
            _, _, abs_freq, w = fft(data, rate)
        else:
            raise ValueError("No fft_data was provided. Missing data and rate. ")
    amp = abs_freq.argmax()
    scaled = abs_freq / abs_freq[amp]
    return scaled[amp] - (np.sum(scaled[:amp]) + np.sum(scaled[amp + 1:]))


def get_dba_level(data: np.ndarray, rate: int, corr_dict: dict[str, float]):
    weighted_signal = A_weight(data, fs=rate)
    rms_value = np.sqrt(np.mean(np.abs(weighted_signal) ** 2))
    result = 20 * np.log10(rms_value)
    xp, corr_interp = interp_correction(corr_dict)
    idx = bisection(xp, result)
    return result + corr_interp[idx]


@freezeargs
@lru_cache
def interp_correction(corr_dict: dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates correction factors represented as a dictionary for key-value pairs. Key represents value
    recorded by the microphone and dedicated value the correction value for dB(A) calculations.

    :param corr_dict: Dictionary (Key-Value pairs) of microphone value and corresponding correction value.
    :return:
    """
    key_list = [float(i) for i in list(corr_dict.keys())]
    val_list = list(corr_dict.values())
    xp = np.linspace(math.floor(min(key_list)), math.ceil(max(key_list)))
    corr_interp = np.interp(xp, key_list, val_list)
    return xp, corr_interp
