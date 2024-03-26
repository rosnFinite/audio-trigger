import logging
import time

import plotly.graph_objs as go
import numpy as np


def create_visualizations(get_event):
    start = time.time()
    logging.debug(f"Creating visualizations for {get_event['dir_path']}, Identifier: {get_event['id']}...")
    logging.debug(f"Plotting waveform for {get_event['dir_path']}...")
    plot_waveform(get_event["numpy"], get_event["dir_path"])
    logging.debug(f"TIME: {time.time() - start}")


def plot_waveform(data, location):
    fig = go.Figure(go.Scatter(x=list(range(len(data))), y=data, mode="lines"))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.write_image(location + "/waveform.png")
    logging.debug(f"Waveform plot saved to {location}/waveform.png")
