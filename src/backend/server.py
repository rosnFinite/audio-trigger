import logging
import sys
import os
import time
from typing import Tuple, Dict, Any, List

from flask import Flask, request, send_from_directory, jsonify, Response
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_cors import CORS

from src.audio.recorder import AudioRecorder
from src.config_utils import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(os.getcwd(), "logs", "server.log"), mode="w")
file_handler.setFormatter(logging.Formatter('%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s'))
logger.addHandler(file_handler)

# solution for path problems using vscode
sys.path.append("D:\\rosef\\audio-trigger")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# variable for the trigger client sid
this = sys.modules[__name__]
# list of connected clients (maximal 2 clients, one web client and one audio trigger client
# each client is represented by a dictionary (key = session ID, value = client type("web", "web_patient", "audio"))
this.connected_clients = []

#TODO: update paths to use path defined in config file
#TODO: update paths to work with new file structure
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
    sid_registered = sid in [client["sid"] for client in this.connected_clients]
    return sid_registered


@app.route("/api/audio-client/devices", methods=["GET"])
def get_devices() -> Tuple[Dict[str, List[Dict[str, str | Any]]], int]:
    """Handles GET requests for the available recording devices. No connected audio trigger client is
    required for this request.

    Returns
    -------
    tuple
        A tuple containing a dictionary with the available devices and an integer status code.
    """
    logger.info(f"GET /audio-client/devices received.")
    device_list = []
    for idx, device in enumerate(AudioRecorder().recording_devices):
        device_list.append({"id": str(idx), "name": device})
    return {"devices": device_list}, 200


@app.route("/api/logs/<log>", methods=["GET"])
def get_logs(log: str) -> tuple[dict[str, str], int] | tuple[Response, int]:
    """Handles GET requests for a log file. No connected audio trigger client is required for this request.

    Parameters
    ----------
    log : str
        The name of the log file to be returned.

    Returns
    -------
    tuple
        A tuple containing the log file content and an integer status code.
    """
    # check if file exists
    if not os.path.exists(os.path.join(os.getcwd(), "logs", f"{log}.log")):
        return {"message": "File not found"}, 404

    with open(os.path.join(os.getcwd(), "logs", f"{log}.log")) as f:
        log_content = f.read()
    return jsonify({"text_content": log_content}), 200


@app.route("/api/recordings/<parent_dir>/<sub_dir>/<img>", methods=["GET"])
def get_image(parent_dir: str, sub_dir: str, img: str) -> tuple[Response, int] | tuple[dict[str, str], int]:
    """Handle GET request for images in the recordings directory either by providing the correct path or the freq/dba
    bin equivalent.

    For example: /api/recordings/[parent]/0_0/[file] will return the image for the first frequency and dba bin.
    Which would be equivalent to providing the path /api/recordings/[parent]/35_55/[file] when 35 and 55 are the
    corresponding dba/freq values.
    """
    logger.info(f"GET /recordings/{parent_dir}/{sub_dir}/{img} received.")
    if img.endswith(".jpg") or img.endswith(".png"):
        return send_from_directory(CONFIG["client_recordings_path"], f"{parent_dir}/{sub_dir}/{img}"), 200
    else:
        logger.critical(f"[GET /recordings/{parent_dir}/{sub_dir}/{img}] Invalid file extension.")
        return {"message": "Provided file extension is not allowed"}, 406


@app.route("/api/recordings/<parent_dir>/<sub_dir>/parsel", methods=["GET"])
def get_parselmouth_stats(parent_dir: str, sub_dir: str) -> tuple[dict[str, str], int] | tuple[Response, int]:
    # check if file exists
    logger.info(f"GET /recordings/{parent_dir}/{sub_dir}/parsel received.")
    file_path = os.path.join(CONFIG["client_recordings_path"], parent_dir, sub_dir, "parsel_stats.txt")
    if not os.path.exists(file_path):
        logger.critical(f"GET /recordings/{parent_dir}/{sub_dir}/parsel File not found: {file_path}")
        return {"message": "File not found"}, 404
    with open(file_path) as f:
        data = f.read()
    return jsonify({"text_content": data}), 200


"""
========================================================================================================================
                                            SOCKET.IO EVENT HANDLERS
========================================================================================================================
"""


@server.on("connect")
def connected():
    """This function is called when a client connects to the server.
    It prints the client's session ID.
    """
    logger.info(f"Client connected: {request.sid}")


@server.on("disconnect")
def disconnected() -> None:
    """This function is called when a client disconnects from the server.
    Will update list of connected clients, emitting the updated list to all clients.
    """
    # TODO: remove after testing
    if len(this.connected_clients) == 0:
        return
    # check if disconnected client was registered and remove it from connected_clients
    for idx, client in enumerate(this.connected_clients):
        if client["sid"] == request.sid:
            this.connected_clients.pop(idx)
            leave_room("client_room", request.sid)
            emit("clients", this.connected_clients, broadcast=True)
    logger.info(f"Client: {request.sid} disconnected.")


@server.on("register")
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
    logger.info(f"Received register event from sid: {request.sid} with data: {data}")
    # check if another client with same type is already connected, except for "web_patient" type
    for client in this.connected_clients:
        if client["type"] == data["type"] and client["type"] != "web_patient":
            # disconnect emitting if that is the case
            disconnect(request.sid)
            logger.debug(f"Client with same type: {data['type']} already exists, disconnecting...")
            return
    # add new client to connected_clients and emit updated list to all clients
    this.connected_clients.append({"sid": request.sid, "type": data["type"]})
    # add client to 'client_room'
    join_room("client_room", request.sid)
    logger.info(f"Client: {request.sid} registered, emitting updated connected_clients list to 'clients'...")
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
    logger.info(f"Received trigger event from sid:{request.sid} with data: {data}")
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
    logger.info(f"Received voice data update event from sid: {request.sid} with data: {data}")
    emit("voice", data, to="client_room", skip_sid=request.sid)


@server.on("settings_update_request")
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
    logger.info(f"Received change settings event from sid: {request.sid} with requested settings: {req_settings}")
    emit("settings_update_request", req_settings, to="client_room", skip_sid=request.sid)


@server.on("settings_update_complete")
def on_settings_changed(updated_settings: dict) -> None:
    """Event handler for when settings are changed. Emitted to by the audio trigger when the settings have changed.

    Parameters
    ----------
    updated_settings: dict
        The updated settings.
    """
    if not check_registration(request.sid):
        return
    logger.info(f"Received settings changed event from sid: {request.sid} with updated settings: {updated_settings}")
    emit("settings_update_complete", updated_settings, to="client_room", skip_sid=request.sid)


@server.on("status_update_request")
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
    logger.info(f"Received change status event from sid:{request.sid} with requested status: {action}")
    emit("status_update_request", action, to="client_room", skip_sid=request.sid)


@server.on("status_update_complete")
def on_status_changed(updated_status: dict) -> None:
    """Event handler for when status is changed. Emitted to by the audio trigger when the status has changed.

    Parameters
    ----------
    updated_status: dict
        The updated status.
    """
    if not check_registration(request.sid):
        return
    logger.info(f"Received status changed event from sid: {request.sid} with updated status: {updated_status}")
    emit("status_update_complete", updated_status, to="client_room", skip_sid=request.sid)


@server.on("remove_recording_request")
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
    logger.info(f"Received remove recording event from sid: {request.sid} with grid location: {grid_location}")
    emit("remove_recording_request", grid_location, to="client_room", skip_sid=request.sid)


def run_server():
    """
    Run the web server on port 5001.
    """
    logger.info("Starting server on port 5001...")
    try:
        server.run(app, port=5001, debug=False, log_output=False)
    except OSError:
        logger.critical("Port 5001 already in use. Server could not be started. Retrying after 10 seconds...")
        time.sleep(10)
        run_server()


if __name__ == '__main__':
    server.run(app, port=5001, debug=True)
