import sys
from typing import Tuple, Dict, Any, List

from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_cors import CORS

from recorder import AudioRecorder

# solution for path problems using vscode
sys.path.append("D:\\rosef\\audio-trigger")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# variable for the trigger client sid
this = sys.modules[__name__]
# list of connected clients (maximal 2 clients, one web client and one audio trigger client
# each client is represented by a dictionary (key = session ID, value = client type("web" or "audio"))
this.connected_clients = []


def check_registration(sid: str) -> bool:
    """Check if the client with the given session ID is registered.

    Parameters
    ----------
    sid : str
        The session ID of the client.

    Returns
    -------
    bool
        True if the client is registered, False otherwise.
    """
    return sid in [client["sid"] for client in this.connected_clients]


@app.route("/api/audio-client/devices", methods=["GET"])
def get_devices() -> Tuple[Dict[str, List[Dict[str, str | Any]]], int]:
    """Handles GET requests for the available recording devices. No connected audio trigger client is
    required for this request.

    Returns
    -------
    tuple
        A tuple containing a dictionary with the available devices and an integer status code.
    """
    device_list = []
    for idx, device in enumerate(AudioRecorder().recording_devices):
        device_list.append({"id": str(idx), "name": device})
    return {"devices": device_list}, 200


@server.on("connect")
def connected():
    """This function is called when a client connects to the server.
    It prints the client's session ID.
    """
    print(f"client has connected: {request.sid}")


@server.on("disconnect")
def disconnected() -> None:
    """This function is called when a client disconnects from the server.
    Will update list of connected clients, emitting the updated list to all clients.
    """
    print("DISCONNECTING CLIENT...")
    # TODO: remove after testing
    if len(this.connected_clients) == 0:
        return
    # check if disconnected client was registered and remove it from connected_clients
    for idx, client in enumerate(this.connected_clients):
        if client["sid"] == request.sid:
            this.connected_clients.pop(idx)
            print(f"removed client:{request.sid} from connected_clients...")
            leave_room("client_room", request.sid)
            emit("clients", this.connected_clients, broadcast=True)
    print(f"client disconnected: {request.sid}")


@server.on("registerClient")
def on_register_client(data: dict) -> None:
    """Event handler for the "registerClient" event. Receives a dictionary containing the client's type "type" and
    session ID "sid". Emitting Client will be added to the connected_clients list if it is not already in the list and
    no other client with the same type exists in the list. If the client is added, the updated list will be emitted
    all clients.

    Parameters
    ----------
    data: dict
        The data received from the client. Dictionary containing the client's type "type" and session ID "sid".
    """
    print("REGISTERING CLIENT...")
    # check if another client with same type is already connected
    for client in this.connected_clients:
        if client["type"] == data["type"]:
            # disconnect emitting if that is the case
            disconnect(request.sid)
            print(f"{request.sid}: client with same type already exists, disconnecting...")
            return
    # add new client to connected_clients and emit updated list to all clients
    this.connected_clients.append({"sid": request.sid, "type": data["type"]})
    # add client to 'client_room'
    join_room("client_room", request.sid)
    print(f"client registered: {request.sid}")
    emit("clients", this.connected_clients, broadcast=True)


@server.on('trigger')
def handle_grid_update(data: dict) -> None:
    """Handle the trigger event and broadcast the data to all except emitter clients. Emitted to by the audio trigger
    when a trigger event occurs.

    Parameters
    ----------
    data: dict
        The data received from the trigger event.
    """
    # only registered clients can emit events
    if not check_registration(request.sid):
        return
    print(f"Received trigger: {data}")
    emit("trigger", data, to="client_room", skip_sid=request.sid)


@server.on("voice")
def handle_voice_update(data: dict) -> None:
    """Event listener when audio information is updated. Emitted to by the audio trigger.

    Parameters
    ----------
    data: dict
        The updated audio information.
    """
    if not check_registration(request.sid):
        return
    print(f"Received audio update: {data}")
    emit("voice", data, to="client_room", skip_sid=request.sid)


@server.on("changeSettings")
def on_settings(req_settings: dict) -> None:
    """Event handler for the "changeSettings" event. Emitted to by the web client when the user wants to change
    settings.

    Parameters
    ----------
    req_settings: dict
        The requested settings to be changed.
    """
    if not check_registration(request.sid):
        return
    print("Settings change request received")
    emit("changeSettings", req_settings, to="client_room", skip_sid=request.sid)


@server.on("settingsChanged")
def on_settings_changed(updated_settings: dict) -> None:
    """Event handler for when settings are changed. Emitted to by the audio trigger when the settings have changed.

    Parameters
    ----------
    updated_settings: dict
        The updated settings.
    """
    if not check_registration(request.sid):
        return
    print("Setting change fulfilled")
    emit("settingsChanged", updated_settings, to="client_room", skip_sid=request.sid)


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
    if not check_registration(request.sid):
        return
    print("Status change request received")
    emit("changeStatus", action, to="client_room", skip_sid=request.sid)


@server.on("statusChanged")
def on_status_changed(updated_status: dict) -> None:
    """Event handler for when status is changed. Emitted to by the audio trigger when the status has changed.

    Parameters
    ----------
    updated_status: dict
        The updated status.
    """
    if not check_registration(request.sid):
        return
    print("status change fulfilled")
    emit("statusChanged", updated_status, to="client_room", skip_sid=request.sid)


@server.on("startTrigger")
def on_start_trigger(device_idx: int) -> None:
    """Event handler for the "startTrigger" event. Emitted to by the web client when the user requests to
    start the trigger.

    Parameters
    ----------
    device_idx: int
        The index of the device to use as the recording and triggering device.
    """
    if not check_registration(request.sid):
        return
    print("trigger process started")
    emit("startTrigger", device_idx, to="client_room", skip_sid=request.sid)


@server.on("removeRecording")
def on_remove_recording(grid_location: dict) -> None:
    """Event handler for the "removeRecording" event. Emitted to by the web client when the user requests to remove a
    recording from the grid. Payload contains the grid location of the recording to be removed {freqBin, dbBin}.

    Parameters
    ----------
    grid_location: dict
        The grid location of the recording to be removed. {freqBin, dbBin}
    """
    if not check_registration(request.sid):
        return
    print("remove recording request received")
    emit("removeRecording", grid_location, to="client_room", skip_sid=request.sid)


if __name__ == '__main__':
    server.run(app, port=5001, debug=True)
