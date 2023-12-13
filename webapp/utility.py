"""
Collection of utility functions.
"""
import functools
import os
from typing import List
from frozendict import frozendict

from recorder import AudioRecorder


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


def load_recording_devices(recorder: AudioRecorder):
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
                   for root, _, files in os.walk(os.path.join(module_dir, "../audio"))
                   for file in files if file.endswith(".wav")]
    return audio_files


def get_calibration_file_names() -> List[dict[str, str]]:
    calib_files = [{"value": os.path.join(root, file), "label": file}
                   for root, _, files in os.walk(os.path.join(module_dir, "../calibration"))
                   for file in files if file.endswith(".json")]
    return calib_files


def bisection(array, value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl


def freezeargs(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped