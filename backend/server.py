import os.path

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from recorder import AudioRecorder

UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = {"json"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*")



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/devices")
def get_devices():
    """return all available recording devices"""
    device_list = []
    for idx, device in enumerate(AudioRecorder().recording_devices):
        device_list.append({"id": str(idx),"name": device})
    return {"devices": device_list}

@app.post("/settings")
def post_settings():
    """handling post request for setting the recording device"""
    data = request.get_json()
    print(data)
    return "Setting successfully received", 200

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


if __name__ == '__main__':
    server.run(app, debug=True, port=5001)


