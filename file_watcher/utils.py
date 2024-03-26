import logging

import plotly.graph_objs as go
import numpy as np


def create_visualizations(get_event):
    logging.debug(f"Creating visualizations for {get_event['dir_path']}...")
    # load .npy into numpy array
    data = np.load(get_event["file_path"])
    logging.debug(f"Plotting waveform for {get_event['dir_path']}...")
    plot_waveform(data, get_event["dir_path"])


def plot_waveform(data, location):
    fig = go.Figure(go.Scatter(x=list(range(len(data))), y=data, mode="lines"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_image(location + "/waveform.png")
    logging.debug(f"Waveform plot saved to {location}/waveform.png")
