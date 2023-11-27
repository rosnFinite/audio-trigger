import dash_mantine_components as dmc
import numpy as np
import scipy
import librosa
from dash import Dash, dcc, State, Input, Output, callback
from dash_iconify import DashIconify
import json
import os
import sys
import plotly.graph_objs as go

from recorder import AudioRecorder

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
# allows to assign variables to it
this.recorder = AudioRecorder(buffer_size=1)

signal_graph_x = [x for x in range(this.recorder.frames.maxlen * this.recorder.chunksize - 1, -1, -1)]

# key is dB(A) value and value is the mean amplitude of the mic
this.db_to_mic_values = dict()
this.mic_value = 0

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
                dmc.Accordion(
                    id="menu",
                    variant="separated",
                    chevronPosition="right",
                    children=[
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Kalibrierung"),
                                dmc.AccordionPanel(
                                    children=[
                                        dmc.Text([
                                            "In gleichen Abstand zu einer Audioquelle das "
                                            "Schallpegelmessgerät und Mikrofon aufstellen. Messungen zu "
                                            "unterschiedlichen dB(A)-Pegeln durchführen und Werte des Mikrofons "
                                            "abspeichern."
                                        ]),
                                        dmc.Space(h=10),
                                        dmc.Text([
                                            "Es werden mindesten 3 Messungen empfohlen. Bei Abschluss der ",
                                            "Kalibrierung wird diese in ein JSON-Format unter ",
                                            dmc.Code("/calibration"),
                                            "gespeichert."
                                        ]),
                                        dmc.Space(h=10),
                                        dmc.Center(
                                            dmc.Paper(
                                                p="md",
                                                withBorder=True,
                                                shadow="md",
                                                children=[
                                                    dmc.Center(
                                                        [
                                                            dmc.NumberInput(
                                                                id="db-value",
                                                                label="dB(A)-Wert",
                                                                description="Gemessen über Schallpegelmessgerät",
                                                                value=40,
                                                                min=0,
                                                                step=5,
                                                                style={"width": 270}
                                                            ),
                                                            dmc.Space(w=20),
                                                            dmc.Stack(
                                                                [
                                                                    dmc.Text("Mikrofonwert", size="md"),
                                                                    dmc.Text(id="microphone-value"),
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    dmc.Space(h=10),
                                                    dmc.Center(
                                                        dmc.ButtonGroup(
                                                            children=[
                                                                dmc.Button(
                                                                    "Speichern",
                                                                    disabled=True,
                                                                    id="save-value-button",
                                                                    color="green"
                                                                ),
                                                                dmc.Button(
                                                                    "Zurücksetzen",
                                                                    id="reset-value-button",
                                                                    color="yellow"
                                                                )
                                                            ]
                                                        )
                                                    ),
                                                ]
                                            ),
                                        ),
                                        dmc.Space(h=20),
                                        dcc.Graph(id='db-microphone-graph'),
                                        dmc.Space(h=20),
                                        dmc.Stack(
                                            [
                                                dmc.Center(
                                                    dmc.TextInput(
                                                        id="calib-filename",
                                                        label="Name der Kalibrierungsdatei",
                                                        style={"width": 400}
                                                    )
                                                ),
                                                dmc.Button(
                                                    "Kalibrierung abspeichern",
                                                    disabled=True,
                                                    id="save-calibration-button",
                                                    color="green"
                                                )
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                            value="calibration"
                        ),
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Trigger"),
                                dmc.AccordionPanel(
                                    [
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
                                                               leftIcon=DashIconify(
                                                                   icon="solar:align-horizonta-spacing-bold"), )
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
                                        dmc.Space(h=20),
                                        dcc.Graph(
                                          id="heatmap"
                                        ),
                                        dmc.Affix(
                                            dmc.Badge(id="note-text",
                                                      size="xl",
                                                      variant="gradient",
                                                      gradient={"from": "teal", "to": "lime", "deg": 105}, ),
                                            position={"bottom": 20, "right": 20}
                                        )
                                    ]
                                )
                            ],
                            value="trigger"
                        )
                    ]
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


# TODO: Bei Wechsel des ausgewählten Mikrofons den aktuellen Stream abbrechen und einen neuen Starten

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
        Output("heatmap", "figure"),
        Output("note-text", "children"),
    ],
    Input("interval-component", "n_intervals"),
    State("menu", "value")
)
def update_live_graph(n_intervals, value):
    if this.recorder.stream is None or value == "calibration":
        return default_plot, default_plot, default_plot, "Kein Audiosignal"
    data = this.recorder.get_audio_data()
    # Plot for audio signal
    signal_fig = go.Figure()
    signal_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    signal_fig.add_trace(go.Scatter(x=signal_graph_x, y=data))
    # Plot for frequencies
    fourier = scipy.fft.fft(data)
    fourier_to_plot = fourier[0:len(fourier) // 2]
    w = np.linspace(0, this.recorder.rate, len(fourier))[0:len(fourier) // 2]
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    abs_freq = np.abs(fourier_to_plot)
    freq_fig.add_trace(go.Scatter(x=w, y=abs_freq))
    amp = abs_freq.argmax()
    freq = w[amp]
    note = librosa.hz_to_note(freq)
    scaled = abs_freq / abs_freq[amp]
    score = scaled[amp] - (np.sum(scaled[:amp]) + np.sum(scaled[amp + 1:]))
    power = np.mean(np.square(data))
    db = 10 * np.log10(power / 1) + 40
    # print(np.mean(np.abs(data)))
    return signal_fig, freq_fig, default_plot, f"{note} ({freq:.2f}) {score:.2f} DB:{np.mean(db):.2f}"


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
    fourier = scipy.fft.fft(data)
    fourier_to_plot = fourier[0:len(fourier) // 2]
    w = np.linspace(0, rate, len(fourier))[0:len(fourier) // 2]
    freq_fig = go.Figure()
    freq_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    abs_freq = np.abs(fourier_to_plot)
    freq_fig.add_trace(go.Scatter(x=w, y=abs_freq))
    amp = abs_freq.argmax()
    freq = w[amp]
    note = librosa.hz_to_note(freq)
    scaled = abs_freq / abs_freq[amp]
    score = scaled[amp] - (np.sum(scaled[:amp]) + np.sum(scaled[amp + 1:]))
    return signal_fig, freq_fig, f"{note} ({freq:.2f}) {score:.2f}"


@callback(
    [
        Output("save-value-button", "disabled"),
        Output("interval-component", "disabled", allow_duplicate=True),
        Output("db-microphone-graph", "figure", allow_duplicate=True),
        Output("start-button", "disabled", allow_duplicate=True),
        Output("stop-button", "disabled", allow_duplicate=True),
    ],
    Input("menu", "value"),
    State("microphone-select", "value"),
    prevent_initial_call=True
)
def hanlde_recording_on_menu_change(menu_value, mic_value):
    fig = go.Figure(go.Scatter(x=[], y=[], mode='lines+markers', ))
    fig.update_layout(
        xaxis_title="dB(A)-Wert",
        yaxis_title="Durchn. Mikrofonamplitude (1 sek.)",
        margin=dict(l=5, r=5, t=5, b=10),
    )
    if menu_value != "calibration":
        return True, True, fig, False, True
    if mic_value == -1:
        print("open but no selection")
        return True, True, fig, False, True
    print("start recording")
    this.recorder.start_stream(input_device_index=mic_value)
    return False, False, fig, True, False


@callback(
    Output("microphone-value", "children"),
    Input("interval-component", "n_intervals")
)
def update_microphone_value_in_calib(n_intervals):
    if this.recorder.stream is None:
        return "Kein Audiosignal"
    data = this.recorder.frames[-1]
    this.mic_value = np.mean(np.abs(data))
    return f"{this.mic_value:.3f}"


@callback(
    [
        Output("db-microphone-graph", "figure", allow_duplicate=True),
        Output("save-calibration-button", "disabled", allow_duplicate=True)
    ],
    Input("save-value-button", "n_clicks"),
    State("db-value", "value"),
    prevent_initial_call=True
)
def on_calibration_value_save_pressed(n_clicks, value):
    if value in this.db_to_mic_values.keys():
        this.db_to_mic_values[value][1] += 1
        this.db_to_mic_values[value][0] = (this.db_to_mic_values[value][0] +
                                           (this.mic_value - this.db_to_mic_values[value][0]) /
                                           this.db_to_mic_values[value][1])
    else:
        this.db_to_mic_values[value] = [this.mic_value, 1]
    print("save")
    fig = go.Figure(go.Scatter(x=list(this.db_to_mic_values.keys()), y=[x[0] for x in this.db_to_mic_values.values()],
                               mode='markers'))
    fig.update_layout(
        xaxis_title="dB(A)-Wert",
        yaxis_title="Durchn. Mikrofonamplitude (1 sek.)",
        margin=dict(l=5, r=5, t=5, b=10),
    )
    return fig, False


@callback(
    [
        Output("db-microphone-graph", "figure", allow_duplicate=True),
        Output("save-calibration-button", "disabled", allow_duplicate=True)
    ],
    Input("reset-value-button", "n_clicks"),
    prevent_initial_call=True
)
def on_calibration_reset_pressed(n_clicks):
    this.db_to_mic_values = dict()
    fig = go.Figure(go.Scatter(x=[], y=[], mode='lines+markers', ))
    fig.update_layout(
        xaxis_title="dB(A)-Wert",
        yaxis_title="Durchn. Mikrofonamplitude (1 sek.)",
        margin=dict(l=5, r=5, t=5, b=10),
    )
    return fig, True


@callback(
    [
        Output("save-calibration-button", "disabled", allow_duplicate=True),
        Output("calib-filename", "error")
    ],

    Input("save-calibration-button", "n_clicks"),
    State("calib-filename", "value"),
    prevent_initial_call=True
)
def on_calibration_save_pressed(n_clicks, filename):
    if filename == "":
        return False, "Bitte Dateiname eingeben"
    with open(f"../calibration/{filename}.json", "w+") as f:
        json.dump(this.db_to_mic_values, f, indent=4)
    return True, ""


@callback(
    [
        Output("start-button", "disabled", allow_duplicate=True),
        Output("stop-button", "disabled", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True)
    ],
    Input("start-button", "n_clicks"),
    State("microphone-select", "value"),
    State("menu", "value"),
    prevent_initial_call=True
)
def start_recording(n_clicks, mic_value, calib_value):
    print("Try start recording")
    print(mic_value)
    if calib_value == "calibration":
        return False, True, True
    if mic_value == -1:
        print("No device selected")
        return False, True, True
    this.recorder.start_stream(input_device_index=mic_value)
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
    Input("audio-file-select", "value"),
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
    Input("microphone-select", "value"),
    State("audio-file-select", "value"),
    prevent_initial_call=True
)
def update_file_selection(mic_select_value, file_select_value):
    if mic_select_value == -1:
        return file_select_value, False
    return "", True


if __name__ == "__main__":
    app.run_server()
