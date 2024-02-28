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

UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = {"json"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

this = sys.modules[__name__]
this.trigger = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/devices")
def get_devices():
    """return all available recording devices"""
    device_list = []
    for idx, device in enumerate(AudioRecorder().recording_devices):
        device_list.append({"id": str(idx), "name": device})
    return {"devices": device_list}


@app.post("/upload-calib")
def upload_calib():
    """handling post request for uploading json calibration files"""
    if "file" not in request.files:
        return "Upload unsuccessful", 400
    file = request.files["file"]
    if file.filename == "":
        return "Upload unsuccessful", 400
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], "calibration.json"))
    return "File successfully uploaded", 200


@server.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(f"client has connected: {request.sid}")


@server.on('trigger')
def handle_grid_update(data):
    """event listener when grid updates"""
    print(f"Received trigger: {data}")
    emit("trigger", data, broadcast=True)


@server.on("voice")
def handle_voice_update(data):
    """event listener when audio information update"""
    print(f"Received audio update: {data}")
    emit("voice", data, broadcast=True)


@server.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("client disconnected")


@server.on("changeSettings")
def on_settings(req_settings):
    print("Settings change request received")
    emit("changeSettings", req_settings, broadcast=True)


@server.on("settingsChanged")
def on_settings_changed(updated_settings):
    print("Setting change fulfilled")
    emit("settingsChanged", updated_settings, broadcast=True)


@server.on("changeStatus")
def on_change_status(req_status):
    print("Status change request received")
    emit("changeStatus", req_status, broadcast=True)


@server.on("statusChanged")
def on_status_changed(updated_status):
    print("status change fulfilled")
    emit("statusChanged", updated_status, broadcast=True)


@server.on("startTrigger")
def on_start_trigger(device_idx):
    print("trigger process started")
    emit("startTrigger", device_idx, broadcast=True)


if __name__ == '__main__':
    server.run(app, port=5001, debug=True)
