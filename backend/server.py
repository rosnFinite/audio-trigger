import os.path
import sys
import eventlet

import socketio
from eventlet import wsgi
from flask import Flask, request, make_response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from recorder import AudioRecorder

# solution for path problems using vscode
sys.path.append("D:\\rosef\\audio-trigger")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")


@app.route("/devices")
def get_devices():
    """Return all available recording devices.

    Returns
    -------
    dict
        A dictionary containing the list of available recording devices. Each device is represented by a dictionary
        with "id" and "name" keys.
    """
    device_list = []
    for idx, device in enumerate(AudioRecorder().recording_devices):
        device_list.append({"id": str(idx), "name": device})
    return {"devices": device_list}


@server.on("connect")
def connected():
    """This function is called when a client connects to the server.
    It prints the client's session ID.
    """
    print(f"client has connected: {request.sid}")


@server.on('trigger')
def handle_grid_update(data: dict) -> None:
    """Handle the trigger event and broadcast the data to all except emitter clients. Emitted to by the audio trigger
    when a trigger event occurs.

    Parameters
    ----------
    data: dict
        The data received from the trigger event.
    """
    print(f"Received trigger: {data}")
    emit("trigger", data, broadcast=True)


@server.on("voice")
def handle_voice_update(data: dict) -> None:
    """Event listener when audio information is updated. Emitted to by the audio trigger.

    Parameters
    ----------
    data: dict
        The updated audio information.
    """
    print(f"Received audio update: {data}")
    emit("voice", data, broadcast=True)


@server.on("disconnect")
def disconnected():
    """This function is called when a client disconnects from the server.
    It prints a message indicating that the client has disconnected.
    """
    print("client disconnected")


@server.on("changeSettings")
def on_settings(req_settings: dict) -> None:
    """Event handler for the "changeSettings" event. Emitted to by the web client when the user wants to change
    settings.

    Parameters
    ----------
    req_settings: dict
        The requested settings to be changed.

    Returns:
        None
    """
    print("Settings change request received")
    emit("changeSettings", req_settings, broadcast=True)


@server.on("settingsChanged")
def on_settings_changed(updated_settings: dict) -> None:
    """Event handler for when settings are changed. Emitted to by the audio trigger when the settings have changed.

    Parameters
    ----------
    updated_settings: dict
        The updated settings.
    """
    print("Setting change fulfilled")
    emit("settingsChanged", updated_settings, broadcast=True)


@server.on("changeStatus")
def on_change_status(action: dict) -> None:
    """Event handler for the "changeStatus" event. Emitted to by the web client when the user requests a status change.
    Receives an action dictionary containing actions for the recorder and trigger.
    Possible actions are: start [trigger & recorder], stop [trigger & recorder], reset [trigger].

    Parameters
    ----------
    action: dict
        The requested action to be performed.
    """
    print("Status change request received")
    emit("changeStatus", action, broadcast=True)


@server.on("statusChanged")
def on_status_changed(updated_status: dict) -> None:
    """Event handler for when status is changed. Emitted to by the audio trigger when the status has changed.

    Parameters
    ----------
    updated_status: dict
        The updated status.
    """
    print("status change fulfilled")
    emit("statusChanged", updated_status, broadcast=True)


@server.on("startTrigger")
def on_start_trigger(device_idx: int) -> None:
    """Event handler for the "startTrigger" event. Emitted to by the web client when the user requests to
    start the trigger.

    Parameters
    ----------
    device_idx: int
        The index of the device to use as the recording and triggering device.
    """
    print("trigger process started")
    emit("startTrigger", device_idx, broadcast=True)


if __name__ == '__main__':
    server.run(app, port=5001, debug=True)
