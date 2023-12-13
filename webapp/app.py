import dash_mantine_components as dmc
import numpy as np
import scipy
from dash import Dash, dcc, State, Input, Output, callback
from dash_iconify import DashIconify
from typing import Tuple
import json
import os
import plotly.graph_objs as go

from components.accordion_panels import CalibrationPanel, DataPanel
from components.information_texts import IntroductionText
from recorder import AudioRecorder
from utility import load_recording_devices, default_plot
from processing.fourier import plot_abs_fft

recorder = AudioRecorder(buffer_size=1, rate=44100)
signal_graph_x = [x for x in range(recorder.frames.maxlen * recorder.chunksize - 1, -1, -1)]

# key is dB(A) value and value is the mean amplitude of the mic
db_to_mic_values = dict()
mic_value = 0

module_path = os.path.abspath(__file__)
module_dir = os.path.dirname(module_path)
audio_files = [{"value": os.path.join(root, file), "label": file}
               for root, _, files in os.walk(os.path.join(module_dir, "../audio"))
               for file in files if file.endswith(".wav")]


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
                          children=IntroductionText
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
                            data=load_recording_devices(recorder),
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
                                CalibrationPanel
                            ],
                            value="calibration"
                        ),
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl("Trigger"),
                                DataPanel
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


"""
=======================================================================================================================
                                        Callbacks for updating graphs
=======================================================================================================================
"""


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
def update_live_graph(n_intervals: int, value: str) -> Tuple[go.Figure, go.Figure, go.Figure, str]:
    """Callback updating live graph of receiving audio input, FFT graph as well as determining main frequency component
    of the signal and corresponding musical note.

    :param n_intervals: Iterations of interval component.
    :param value: Currently selected accordion menu "calibration" or "trigger".
    :return:
    """
    if recorder.stream is None or value == "calibration":
        return default_plot, default_plot, default_plot, "Kein Audiosignal"
    data = recorder.get_audio_data()
    data = data.copy()
    # Plot for audio signal
    signal_fig = go.Figure()
    signal_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    signal_fig.add_trace(go.Scatter(x=signal_graph_x, y=data))
    # Plot for frequencies
    freq_fig, note, score = plot_abs_fft(data, recorder.rate)
    return signal_fig, freq_fig, default_plot, f"{note} [{score:.2f}]"


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
def update_file_graph(n_clicks: int, file_select_value: str) -> Tuple[go.Figure, go.Figure, str]:
    """Callback updating graph of audio input provided via selected WAV file, FFT graph as well as determining main
    frequency component of the signal and corresponding musical note.

    :param n_clicks: Number of clicks performed on the "load-file-button". Triggers this callback.
    :param file_select_value: Audio file selected in dropdown menu
    :return:
    """
    rate, data = scipy.io.wavfile.read(os.path.join("../audio", file_select_value))
    signal_fig = go.Figure()
    signal_fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=10),
    )
    signal_fig.add_trace(go.Scatter(x=signal_graph_x, y=data))
    # Plot for frequencies
    freq_fig, note, score = plot_abs_fft(data, recorder.rate)
    return signal_fig, freq_fig, f"{note} [{score:.2f}]"


@callback(
    Output("microphone-value", "children"),
    Input("interval-component", "n_intervals")
)
def update_microphone_value_in_calib(n_intervals: int) -> str:
    """Callback for updating microphone input values displayed in calibration menu.

    :param n_intervals: Iterations of interval component.
    :return:
    """
    if recorder.stream is None:
        return "Kein Audiosignal"
    data = recorder.frames[-1]
    mic_value = np.mean(np.abs(data))
    return f"{mic_value:.3f}"


"""
=======================================================================================================================
                                        Callbacks for menu selections
=======================================================================================================================
"""


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
def hanlde_recording_on_menu_change(menu_value: str, mic_value: int) -> Tuple[bool, bool, go.Figure, bool, bool]:
    """Callback for automatic starting and stopping of audio input streams when opening and closing accordion panels.
    Will additionally update states of other components such as the ButtonGroup for controlling the trigger.

    :param menu_value: Currently opened accordion panel.
    :param mic_value: Index of selected input device.
    :return:
    """
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
    recorder.start_stream(input_device_index=mic_value)
    return False, False, fig, True, False


@callback(
    Output("microphone-select", "value", allow_duplicate=True),
    Output("start-button", "disabled"),
    Input("audio-file-select", "value"),
    State("microphone-select", "value"),
    prevent_initial_call=True
)
def update_mic_selection(file_select_value: str, mic_select_value: int) -> Tuple[int, bool]:
    """Callback handling whether a microphone was selected. Ensures that either an audio file or
    microphone is selected but not both.

    :param file_select_value: Selected audio file, empty string if none is selected.
    :param mic_select_value: Selected microphone, -1 if none is selected.
    :return:
    """
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
def update_file_selection(mic_select_value: int, file_select_value: str) -> Tuple[str, bool]:
    """Callback handling whether an audio file was selected. Ensures that either an audio file or
    microphone is selected but not both.

    :param mic_select_value: Selected microphone, -1 if none is selected.
    :param file_select_value: Selected audio file, empty string if none is selected.
    :return:
    """
    if mic_select_value == -1:
        return file_select_value, False
    return "", True


"""
=======================================================================================================================
                                        Callbacks for pressed buttons
=======================================================================================================================
"""


@callback(
    [
        Output("db-microphone-graph", "figure", allow_duplicate=True),
        Output("save-calibration-button", "disabled", allow_duplicate=True)
    ],
    Input("save-value-button", "n_clicks"),
    State("db-value", "value"),
    prevent_initial_call=True
)
def on_calibration_value_save_pressed(n_clicks: int, value: float) -> Tuple[go.Figure, bool]:
    """Callback for saving microphone and corresponding selected db value for microphone calibration.

    :param n_clicks: Number of clicks performed on the "save-value-button". Triggers this callback.
    :param value: Selected reference db value.
    :return:
    """
    if value in db_to_mic_values.keys():
        db_to_mic_values[value][1] += 1
        db_to_mic_values[value][0] = (db_to_mic_values[value][0] +
                                      (mic_value - db_to_mic_values[value][0]) /
                                      db_to_mic_values[value][1])
    else:
        db_to_mic_values[value] = [mic_value, 1]
    print("save")
    fig = go.Figure(go.Scatter(x=list(db_to_mic_values.keys()), y=[x[0] for x in db_to_mic_values.values()],
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
def on_calibration_reset_pressed(n_clicks: int) -> Tuple[go.Figure, bool]:
    """Callback for resetting saved calibration data.

    :param n_clicks: Number of clicks performed on the "reset-value-button". Triggers this callback.
    :return:
    """
    db_to_mic_values = dict()
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
def on_calibration_save_pressed(n_clicks: int, filename: str) -> Tuple[bool, str]:
    """Callback for writing collected calibration data into a JSON file inside /calibration.

    :param n_clicks: Number of clicks performed on the "save-value-button". Triggers this callback.
    :param filename: Name of the file in which to store calibration data.
    :return:
    """
    if filename == "":
        return False, "Bitte Dateiname eingeben"
    with open(f"../calibration/{filename}.json", "w+") as f:
        json.dump(db_to_mic_values, f, indent=4)
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
def start_recording(n_clicks: int, mic_value: int, calib_value: str) -> Tuple[bool, bool, bool]:
    """Callback handling starting of audio input stream depending on currently opened panel and recoder status.
    Will enable or disable interval component as well as play, stop and trigger button depending on recoder status.

    :param n_clicks: Number of clicks performed on the "start-button". Triggers this callback.
    :param mic_value: Selected microphone
    :param calib_value: Currently opened panel
    :return:
    """
    print("Try start recording")
    print(mic_value)
    if calib_value == "calibration":
        return False, True, True
    if mic_value == -1:
        print("No device selected")
        return False, True, True
    recorder.start_stream(input_device_index=mic_value)
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
def stop_recording(n_clicks: int) -> Tuple[bool, bool, bool]:
    """Callback handling stopping of audio input stream.
    Will enable or disable interval component as well as play and stop button.

    :param n_clicks: Number of clicks performed on the "stop-button". Triggers this callback.
    :return:
    """
    print("stop recording")
    recorder.stop_stream()
    return False, True, True


if __name__ == "__main__":
    app.run_server()
