from typing import Optional, Tuple

import numpy as np
import parselmouth

from audio.processing.frequency import fourier_transform


def calc_quality_score(data: Optional[np.ndarray] = None,
                       rate: Optional[int] = None,
                       abs_freq: Optional[np.ndarray] = None) -> float:
    """Calculates quality score denoting the strength of strongest frequency compared to other frequencies.
    Values closer to 0 denote higher quality/strength of the mainly detected frequency.

    Parameters
    ----------
    data : np.ndarray
        Numpy array of the recorded audio data.
    rate : int
        Sampling rate of the provided audio data.
    abs_freq : Optional[np.ndarray]
        The absolute frequency domain of the provided audio data.

    Returns
    -------
    float
        Quality score of the provided audio data.
    """
    if abs_freq is None:
        if data is not None and rate is not None:
            freq_domain, _ = fourier_transform(data, rate)
            abs_freq = np.abs(freq_domain)
        else:
            raise ValueError("No fft_data was provided. Missing data and rate. ")
    amp = abs_freq.argmax()
    scaled = abs_freq / abs_freq[amp]
    return abs(scaled[amp] - (np.sum(scaled[:amp]) + np.sum(scaled[amp + 1:])))


def calc_pitch_score(data: np.ndarray, rate: int) -> Tuple[float, float]:
    """Calculates the pitch score of the provided audio data. The pitch score is a value between 0 and 1, where 1
    denotes a high pitch consistency and 0 a low pitch consistency. The pitch score is calculated by the standard
    deviation of the detected pitch values. The pitch value is the mean of the detected pitch values.

    Parameters
    ----------
    data : np.ndarray
        Numpy array of the recorded audio data.
    rate : int
        Sampling rate of the provided audio data.

    Returns
    -------
    Tuple[float, float]
        Tuple containing the pitch score and the mean pitch value.
    """
    sound = parselmouth.Sound(data, sampling_frequency=rate)

    # intensity score via standard deviation 1 = high consistency, 0 = low consistency over time
    intensity = sound.to_intensity(time_step=0.01)
    intensity_std = np.std(intensity.values)
    intensity_score = 1 / (1+np.std(intensity.values))

    # frequency score
    pitch = sound.to_pitch(time_step=0.01)
    pitch_values = [val[0] for val in pitch.selected_array]
    pitch_std = np.std(pitch_values)
    pitch_score = 1 / (pitch_std+1)

    return round((intensity_score + pitch_score) / 2, 4), np.mean(pitch_values)