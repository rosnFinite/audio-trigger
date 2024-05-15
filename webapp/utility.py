"""
Collection of utility functions.
"""
import os

import librosa
import numpy as np
import plotly.graph_objs as go
from typing import List, Optional, Tuple

from src.audio.processing.scoring import calc_quality_score, fourier_transform

module_path = os.path.abspath(__file__)
module_dir = os.path.dirname(module_path)


default_plot = {
    "layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
        "annotations": [
            {
                "text": "Keine Daten zum Visualisieren",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 20
                }
            }
        ]
    }
}


def load_recording_devices(recorder):
    """Functionality to receive all audio devices connected to the current system.

    :return: A list of dictionaries each containing the keys 'value' and 'label'
    """
    info = recorder.p.get_host_api_info_by_index(0)
    data = []
    for i in range(0, info.get('deviceCount')):
        if (recorder.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            data.append({
                "value": i,
                "label": recorder.p.get_device_info_by_host_api_device_index(0, i).get('name')
            })
    return data


def get_audio_file_names() -> List[dict[str, str]]:
    audio_files = [{"value": os.path.join(root, file), "label": file}
                   for root, _, files in os.walk(os.path.join(module_dir, "../src/audio"))
                   for file in files if file.endswith(".wav")]
    return audio_files


def get_calibration_file_names() -> List[dict[str, str]]:
    calib_files = [{"value": os.path.join(root, file), "label": file}
                   for root, _, files in os.walk(os.path.join(module_dir, "../calibration"))
                   for file in files if file.endswith(".json")]
    return calib_files


def get_dominant_note(data: Optional[np.ndarray] = None,
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
    freq = get_dominant_freq(data, rate, abs_freq, w)
    try:
        note = librosa.hz_to_note(freq)
    except OverflowError:
        #TODO excetion if freq == 0
        return None
    return note


def plot_abs_fft(data: np.ndarray, rate: int) -> Tuple[go.Figure, str, float]:
    """Performs FFT on provided data and returns a go.Figure visualizing the frequency domain.

    :param data: Numpy array of the recorded audio data.
    :param rate: Sampling rate of the recorder.
    :return:
    """
    freq_domain, _ = fourier_transform(data, rate)
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    abs_freq = np.abs(freq_domain)
    w = np.linspace(0, rate, len(freq_domain))[0:len(freq_domain) // 2]
    freq_fig.add_trace(go.Scatter(x=w, y=abs_freq))
    note = get_dominant_note(abs_freq=abs_freq, w=w)
    score = calc_quality_score(abs_freq=abs_freq)
    return freq_fig, note, score

