from typing import Optional, Tuple

import numpy as np
import parselmouth

from src.voice_analysis.postprocessing.frequency import fourier_transform


def calc_quality_score(data: Optional[np.ndarray] = None,
                       rate: Optional[int] = None,
                       abs_freq: Optional[np.ndarray] = None) -> float:  # pragma: no cover
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


def calc_pitch_score(data: Optional[np.ndarray] = None,
                     rate: Optional[int] = None,
                     sound: Optional[parselmouth.Sound] = None,
                     freq_floor: Optional[float] = None,
                     freq_ceiling: Optional[float] = None) -> Tuple[float, float]:
    """Calculates the pitch score of the provided audio data. The pitch score is a value between 0 and 1, where 1
    denotes a high pitch consistency and 0 a low pitch consistency. The pitch score is calculated by the standard
    deviation of the detected pitch values. The pitch value is the mean of the detected pitch values.

    Either a parselmouth sound object or the audio data and rate must be provided.

    Parameters
    ----------
    data : Optional[np.ndarray]
        Numpy array of the recorded audio data.
    rate : Optional[int]
        Sampling rate of the provided audio data.
    sound : Optional[parselmouth.Sound]
        Parselmouth sound object.
    freq_floor : Optional[float]
        Lower frequency limit for pitch detection. [required for sound object]
    freq_ceiling : Optional[float]
        Upper frequency limit for pitch detection. [required for sound object]

    Returns
    -------
    Tuple[float, float]
        Tuple containing the pitch score and the mean pitch value.
    """
    if sound is None:
        if data is None or rate is None:
            raise ValueError("No data and rate provided. ")
        sound = parselmouth.Sound(data, sampling_frequency=rate)
    else:
        if freq_floor is None or freq_ceiling is None:
            raise ValueError("No frequency bounds provided.")

    # intensity score via standard deviation 1 = high consistency, 0 = low consistency over time
    intensity = sound.to_intensity(time_step=0.001)
    intensity_std = np.std(intensity.values)
    intensity_score = 1 / (1 + intensity_std)

    # frequency score
    pitch = sound.to_pitch(time_step=0.001, pitch_floor=freq_floor, pitch_ceiling=freq_ceiling)
    pitch_values = [val[0] for val in pitch.selected_array]
    pitch_std = np.std(pitch_values)
    pitch_score = 1 / (1 + pitch_std)

    return round((intensity_score + pitch_score) / 2, 4), np.mean(pitch_values)
