"""
Collection of utility functions.
"""
from recorder import AudioRecorder


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
