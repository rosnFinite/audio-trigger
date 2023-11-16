import dash_mantine_components as dmc
import numpy as np
import scipy
import librosa
from dash import Dash, dcc, State, Input, Output, callback
from dash_iconify import DashIconify
import os
import sys
import plotly.graph_objs as go
from timeit import default_timer as timer
from recorder import AudioRecorder

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
# allows to assign variables to it
this.recorder = AudioRecorder(buffer_size=1)

signal_graph_x = [x for x in range(this.recorder.frames.maxlen * this.recorder.chunksize - 1, -1, -1)]

module_path = os.path.abspath(__file__)
module_dir = os.path.dirname(module_path)
audio_files = [{"value": os.path.join(root, file), "label": file}
               for root, _, files in os.walk(os.path.join(module_dir, "../audio"))
               for file in files if file.endswith(".wav")]

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


def load_recording_devices():
    """Functionality to receive all audio devices connected to the current system.

    :return: A list of dictionaries each containing the keys 'value' and 'label'
    """
    info = this.recorder.p.get_host_api_info_by_index(0)
    data = []
    for i in range(0, info.get('deviceCount')):
        if (this.recorder.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            data.append({
                "value": i,
                "label": this.recorder.p.get_device_info_by_host_api_device_index(0, i).get('name')
            })
    return data


app = Dash(
    __name__,
    external_stylesheets=[
        # include google fonts
        "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;900&display=swap"
    ],
)

app.layout = dmc.MantineProvider(
    id="mantine-provider",
    theme={
        "fontFamily": "'Inter', sans-serif",
        "colorScheme": "light",
        "primaryColor": "indigo",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    },
    inherit=True,
    withGlobalStyles=True,
    withNormalizeCSS=True,
    children=[
        dmc.Header(
            height=80,
            p="md",
            children=[
                dmc.Group(
                    position="apart",
                    align="flex-start",
                    children=[
                        dmc.Text("Prototyp: Audiovisualisierung (Trigger)", style={"fontSize": 30}),
                        dmc.ActionIcon(DashIconify(icon="codicon:color-mode", width=20), id="theme-icon", size="xl")
                    ]
                )
            ],
            style={"color": "#657fef"}
        ),
        dmc.Container(
            fluid=True,
            py="xl",
            children=[
                dmc.Paper(radius="sm",
                          withBorder=True,
                          p="md",
                          shadow="md",
                          children=[dmc.Text(
                              [
                                  "Hierbei handelt es sich um einen Prototyp zur ",
                                  "Visualisierung einkommender Audiosignale, ",
                                  "als auch für übergebene WAV-Dateien. Hierzu muss im ",
                                  "Folgenden entweder ein gewünschtes Eingabegerät ausgewählt werden oder ",
                                  "aber eine WAV-Datei übegeben werden."
                              ]
                          ),
                              dmc.Space(h=10),
                              dmc.Text(
                                  [
                                      "Je nach OS kann es vorkommen, dass gewisse Eingabegeräte keine ",
                                      "passende Ausgabe liefern. In diesem Fall muss das innerhalb des OS als Standard"
                                      "festgelegte Gerät ausgewäht werden."
                                      "(Der Graph wird in ",
                                      "einem festgelegten Intervall von 200ms aktualisiert)"
                                  ]
                              ),
                              dmc.Space(h=10),
                              dmc.Text(
                                  [
                                      "Wurde eine Audiodatei übergeben so wird das Signal dieser äquivalent zum ",
                                      "Audiostreaming unterhalb visualisiert. Damit eine WAV-Datei ausgewählt ",
                                      "werden kann, muss sie im Projektordner unter ",
                                      dmc.Code("/audio"),
                                      " abgelegt werden."
                                  ]),
                              dmc.Space(h=10),
                              dmc.Text(
                                  [
                                      "Falls bereits ein Graph erstellt wurde und die Audiodatei oder ",
                                      "das Eingabegerät geändert werden, so wird der Graph dadurch überschrieben."
                                  ],
                                  color="blue"
                              )
                          ],
                          ),
                dmc.Space(h=10),
                dmc.Group(
                    position="center",
                    children=[
                        dmc.Select(
                            label="Aufnahmegerät",
                            placeholder="auswählen",
                            clearable=True,
                            id="microphone-select",
                            value=-1,
                            data=load_recording_devices(),
                            style={"width": 300},
                        ),
                        dmc.Select(
                            label="Audiodatei",
                            placeholder="auswählen",
                            clearable=True,
                            id="audio-file-select",
                            value="",
                            data=audio_files,
                            style={"width": 300}
                        ),
                        dmc.Button(
                            "Laden",
                            id="load-file-button",
                            disabled=True,
                            color="green",
                            style={"marginTop": 24}
                        )
                    ]
                ),
                dmc.Space(h=10),
                dmc.Center(
                    dmc.ButtonGroup(
                        children=[
                            dmc.Button("Start",
                                       id="start-button",
                                       size="lg",
                                       color="green",
                                       disabled=True,
                                       leftIcon=DashIconify(icon="solar:play-outline"), ),
                            dmc.Button("Stop",
                                       id="stop-button",
                                       size="lg",
                                       color="red",
                                       disabled=True,
                                       leftIcon=DashIconify(icon="solar:stop-outline"), ),
                            dmc.Button("Trigger",
                                       id="trigger-button",
                                       size="lg",
                                       color="yellow",
                                       disabled=True,
                                       leftIcon=DashIconify(icon="solar:align-horizonta-spacing-bold"),)
                        ]
                    )
                ),
                dmc.Space(h=20),
                dcc.Graph(
                    id='signal-graph'
                ),
                dmc.Space(h=20),
                dcc.Graph(
                    id='frequency-graph'
                ),
                dmc.Affix(
                    dmc.Badge(id="note-text",
                              size="xl",
                              variant="gradient",
                              gradient={"from": "teal", "to": "lime", "deg": 105}, ),
                    position={"bottom": 20, "right": 20}
                )
            ]
        ),
        dcc.Interval(
            id='interval-component',
            disabled=True,
            interval=200,  # in milliseconds
            n_intervals=0
        ),
    ],
)


@callback(
    Output("mantine-provider", "theme"),
    Input("theme-icon", "n_clicks"),
    State("mantine-provider", "theme"),
    prevent_initial_call=True
)
def change_color_scheme(n_clicks, theme):
    if theme["colorScheme"] == "dark":
        theme["colorScheme"] = "light"
    else:
        theme["colorScheme"] = "dark"
    return theme


@callback(
    [
        Output("signal-graph", "figure"),
        Output("frequency-graph", "figure"),
        Output("note-text", "children"),
    ],
    Input("interval-component", "n_intervals")
)
def update_live_graph(n_intervals):
    if this.recorder.stream is None:
        return default_plot, default_plot, "Kein Audiosignal"
    data = this.recorder.get_audio_data()
    # Plot for audio signal
    signal_fig = go.Figure()
    signal_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    signal_fig.add_trace(go.Scatter(x=signal_graph_x, y=data))
    # Plot for frequencies
    len_data = len(data)
    channel_data = np.zeros(2 ** (int(np.ceil(np.log2(len_data)))))
    channel_data[0:len_data] = data
    fourier = scipy.fft.fft(channel_data)
    fourier_to_plot = fourier[0:len(fourier) // 2]
    w = np.linspace(0, this.recorder.rate, len(fourier))[0:len(fourier) // 2]
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    freq_fig.add_trace(go.Scatter(x=w, y=np.abs(fourier_to_plot)))
    amp = fourier_to_plot.argmax()
    freq = w[amp]
    note = librosa.hz_to_note(freq)
    return signal_fig, freq_fig, f"{note} ({freq:.2f})"


@callback(
    [
        Output("start-button", "disabled", allow_duplicate=True),
        Output("stop-button", "disabled", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True)
    ],
    Input("start-button", "n_clicks"),
    State("microphone-select", "value"),
    prevent_initial_call=True
)
def start_recording(n_clicks, value):
    print("Try start recording")
    print(value)
    if value == -1:
        print("No device selected")
        return False, True, True
    this.recorder.start_stream(input_device_index=value)
    return True, False, False


@callback(
    [
        Output("start-button", "disabled", allow_duplicate=True),
        Output("stop-button", "disabled", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True)
    ],
    Input("stop-button", "n_clicks"),
    prevent_initial_call=True
)
def stop_recording(n_clicks):
    print("stop recording")
    this.recorder.stop_stream()
    return False, True, True


@callback(
    Output("microphone-select", "value", allow_duplicate=True),
    Output("start-button", "disabled"),
    Input("audio-file-select","value"),
    State("microphone-select", "value"),
    prevent_initial_call=True
)
def update_mic_selection(file_select_value, mic_select_value):
    if file_select_value == "":
        return mic_select_value, False
    return -1, True


@callback(
    Output("audio-file-select", "value", allow_duplicate=True),
    Output("load-file-button", "disabled"),
    Input("microphone-select","value"),
    State("audio-file-select", "value"),
    prevent_initial_call=True
)
def update_file_selection(mic_select_value, file_select_value):
    if mic_select_value == -1:
        return file_select_value, False
    return "", True


@callback(
    [
        Output("signal-graph", "figure", allow_duplicate=True),
        Output("frequency-graph", "figure", allow_duplicate=True),
        Output("note-text", "children", allow_duplicate=True),
    ],
    Input("load-file-button", "n_clicks"),
    State("audio-file-select", "value"),
    prevent_initial_call=True
)
def update_file_graph(n_clicks, file_select_value):
    rate, data = scipy.io.wavfile.read(os.path.join("../audio", file_select_value))
    signal_fig = go.Figure()
    signal_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    signal_fig.add_trace(go.Scatter(x=signal_graph_x, y=data))
    # Plot for frequencies
    len_data = len(data)
    channel_data = np.zeros(2 ** (int(np.ceil(np.log2(len_data)))))
    channel_data[0:len_data] = data
    fourier = scipy.fft.fft(channel_data)
    fourier_to_plot = fourier[0:len(fourier) // 2]
    w = np.linspace(0, rate, len(fourier))[0:len(fourier) // 2]
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    freq_fig.add_trace(go.Scatter(x=w, y=np.abs(fourier_to_plot)))
    amp = fourier_to_plot.argmax()
    freq = w[amp]
    note = librosa.hz_to_note(freq)
    return signal_fig, freq_fig, note


if __name__ == "__main__":
    app.run_server()
