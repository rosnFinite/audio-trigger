import os.path

from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

UPLOAD_FOLDER = "./tmp"
ALLOWED_EXTENSIONS = {"json"}

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
server = SocketIO(app, cors_allowed_origins="*")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


@server.on('grid-update')
def handle_grid_update(data):
    """event listener when grid updates"""
    print(f"Received grid update: {data}")


@server.on("voice-update")
def handle_voice_update(data):
    """event listener when audio information update"""
    print(f"Received audio update: {data}")


@server.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    print("client disconnected")
    emit("disconnect", f"user {request.sid} disconnected", broadcast=True)


if __name__ == '__main__':
    server.run(app, debug=True, port=5001)


