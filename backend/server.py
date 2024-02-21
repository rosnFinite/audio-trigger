import os.path
import sys
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# solution for path problems using vscode
sys.path.append("D:\\rosef\\audio-trigger")

from recorder import Trigger

UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = {"json"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

trigger = Trigger(buffer_size=0.2, rec_destination="test")
devices = trigger.recording_devices
selected_device = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/input-devices")
def input_devices_get():
    """return the available audio input devices on local system"""
    data = {"devices": devices}
    return jsonify(data)


@app.route("/input-device", methods=["GET", "POST"])
def input_device():
    """returns currently selected input device or sets a new input device"""
    if request.method == "GET":
        if trigger.recording_device is None:
            return jsonify({"device": ""})
        return jsonify({"device": devices[trigger.recording_device]})
    else:
        # post only expects application/json with key "device" and int as value
        # simple verification for correct key
        if "device" not in list(request.json.keys()):
            return "Missing required key 'device' in request", 400
        if type(request.json["device"]) is not int:
            return "Invalid type of value for key 'device'", 400
        return "Device successfully selected", 200


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


@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(request.sid)
    print("client has connected")
    emit("connect", {"data": f"id: {request.sid} is connected"})


@socketio.on('data')
def handle_message(data):
    """event listener when client types a message"""
    print("data: ", str(data))
    emit("data", {'data': data, 'id': request.sid}, broadcast=True)


@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("user disconnected")
    emit("disconnect", f"user {request.sid} disconnected", broadcast=True)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
