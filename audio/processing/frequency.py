from typing import Tuple

import numpy as np
import scipy


def fourier_transform(data: np.ndarray, rate: int) -> Tuple[np.ndarray, float]:
    """Performs FFT on provided data and returns the frequency domain and the dominant frequency.

    Parameters
    ----------
    data : np.ndarray
        Numpy array of the recorded audio data.
    rate : int
        Sampling rate of the provided audio data.

    Returns
    -------
    freq_domain : np.ndarray
        The frequency domain of the provided audio data.
    dominant_freq : float
        The dominant frequency of the provided audio data.
    """
    freq_domain = scipy.fft.rfft(data)
    abs_freq = np.abs(freq_domain)
    w = np.linspace(0, rate, len(freq_domain))[0:len(freq_domain) // 2]

    amp = abs_freq.argmax()
    dominant_freq = w[amp]
    return freq_domain, dominant_freq

