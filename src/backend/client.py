import os
import sys
import time

import socketio
import logging

from typing import List, Dict

from src.audio.recorder import AudioTriggerRecorder
from src.config_utils import CONFIG
from src.exception_manager import emit_exception

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(os.getcwd(), "logs", "client.log"), mode="w")
file_handler.setFormatter(logging.Formatter('%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s'))
logger.addHandler(file_handler)

client = socketio.Client(engineio_logger=False)
this = sys.modules[__name__]
this.trigger_recorder = None


@client.on("connect")
def on_connect() -> None:
    """This function is called when the client successfully connects to the server.
    It prints a message indicating that the connection has been established.
    """
    logger.info("Connection to websocket server established. Registering as audio trigger client...")
    client.emit("register", {"type": "audio"})


@client.on("clients")
def on_clients(clients: list) -> None:
    """Event handler for the "clients" event.

    Parameters
    ----------
    clients: List[Dict[str]]
        The data received from the audio trigger client. Contains the client's session ID.
    """
    logger.debug(f"Received clients event with connected client: {clients}", )


@client.on("settings_update_request")
def on_settings_change(settings: dict) -> None:
    """Function to handle settings change event.

    Parameters
    ----------
    settings : dict
        Dictionary containing the updated settings.
    """
    logger.info("Change settings event received. Creating new trigger instance..")
    # build path for recordings/measurements  <patient>_<timestamp> if patient is set
    if settings["patient"] == "":
        destination = os.path.join(CONFIG["client_recordings_path"], f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    else:
        destination = os.path.join(CONFIG["client_recordings_path"], f"{settings['patient']}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")

    this.trigger_recorder = AudioTriggerRecorder(rec_destination=destination,
                                                 min_score=settings["min_score"],
                                                 retrigger_percentage_improvement=settings["retrigger_percentage_improvement"],
                                                 semitone_bin_size=settings["frequency"]["steps"],
                                                 freq_bounds=(settings["frequency"]["lower"], settings["frequency"]["upper"]),
                                                 dba_bin_size=settings["db"]["steps"],
                                                 dba_bounds=(settings["db"]["lower"], settings["db"]["upper"]),
                                                 buffer_size=settings["buffer_size"],
                                                 channels=1 if settings["mono"] else 2,
                                                 rate=settings["sampling_rate"],
                                                 chunk_size=settings["chunk_size"],
                                                 socket=client)
    settings["save_location"] = this.trigger_recorder.rec_destination
    if settings["device"] == -1:
        this.trigger_recorder.recording_device = 1
        #TODO: Ändern für automatische Auswahl des iMic-Microphone
    else:
        this.trigger_recorder.recording_device = settings["device"]
    settings["status"] = "ready"

    logger.debug("Verifying number of channels for selected audio device...")
    # try opening audio stream to verify number channels are supported
    if this.trigger_recorder.try_open_stream(input_device_index=this.trigger_recorder.recording_device):
        logger.debug("Audio device supports the required number of channels.")
        logger.debug("Emitting changed settings to server...")
        client.emit("settings_update_complete", settings)
    else:
        this.trigger_recorder = None
        logger.error("Audio device does not support the required number of channels.")
        emit_exception(client, exception_level="critical", exception_title="Prozess konnte nicht gestartet werden!",
                       exception_description="Aufnahmegerät unterstützt NICHT die eingestellte "
                                             "Anzahl an Kanälen. Gerät oder Kanalanzahl anpassen.")


@client.on("status_update_request")
def on_status_update(action: dict) -> None:
    """Function to handle status updates. Receives an action dictionary containing actions for the recorder and trigger.
    Possible actions are: start [trigger & recorder], stop [trigger & recorder], reset [trigger].

    Parameters
    ----------
    action : dict
        Dictionary containing the action trigger.
    """
    logger.info(f"Change status event received: {action}")
    if this.trigger_recorder is None:
        logger.error("No trigger instance available. Cannot perform action.")
        return
    if action["trigger"] == "start":
        # only do smth if trigger is not already running
        if not this.trigger_recorder.stream_thread_is_running:
            logger.info(f"Starting trigger, device: {this.trigger_recorder.recording_device}")
            this.trigger_recorder.start_trigger()
            client.emit("status_update_complete", {"status": "running", "save_location": this.trigger_recorder.rec_destination})
    if action["trigger"] == "stop":
        # only do smth if trigger is currently running
        if this.trigger_recorder.stream_thread_is_running:
            this.trigger_recorder.stop_trigger()
            logger.info("AudioTriggerRecorder stopped.")
            client.emit("status_update_complete", {"status": "ready", "save_location": this.trigger_recorder.rec_destination})
    if action["trigger"] == "reset":
        if not this.trigger_recorder.stream_thread_is_running:
            this.trigger_recorder.rec_destination = this.trigger_recorder.trigger.reset_field()
            logger.info("AudioTriggerRecorder reset.")
            client.emit("status_update_complete", {"status": "reset", "save_location": this.trigger_recorder.rec_destination})


@client.on("remove_recording_request")
def on_remove_recording(grid_location: dict) -> None:
    """Function to handle the "removeRecording" event. Removes the recording at the specified grid location.

    Parameters
    ----------
    grid_location : dict
        Dictionary containing the grid location of the recording to be removed. {freqBin, dbBin}
    """
    this.trigger_recorder.voice_field.field[grid_location["dbaBin"]][grid_location["freqBin"]] = None
    logger.info(f"Removed recording at grid location: {grid_location}")


def run_client():
    client.connect("http://localhost:5001")
    client.wait()


if __name__ == "__main__":
    try:
        client.connect("http://localhost:5001")
        client.wait()
    except KeyboardInterrupt:
        client.disconnect()
        sys.exit(0)