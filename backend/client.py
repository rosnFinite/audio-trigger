import os
import sys
import time

import socketio
import logging

from typing import List, Dict

from audio.recorder import Trigger

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(os.getcwd(), "logs", "client.log"), mode="w")
file_handler.setFormatter(logging.Formatter('%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s'))
logger.addHandler(file_handler)

client = socketio.Client(engineio_logger=False)
this = sys.modules[__name__]
this.trigger = None


@client.on("connect")
def on_connect() -> None:
    """This function is called when the client successfully connects to the server.
    It prints a message indicating that the connection has been established.
    """
    logger.info("Connection to websocket server established. Registering as audio trigger client...")
    client.emit("registerClient", {"type": "audio"})


@client.on("clients")
def on_clients(clients: list) -> None:
    """Event handler for the "clients" event.

    Parameters
    ----------
    clients: List[Dict[str]]
        The data received from the audio trigger client. Contains the client's session ID.
    """
    logger.debug(f"Received clients event with connected client: {clients}", )


@client.on("changeSettings")
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
        destination = os.path.join(os.getcwd(), "backend", "recordings", f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")
    else:
        destination = os.path.join(os.getcwd(), "backend", "recordings", f"{settings['patient']}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}")

    this.trigger = Trigger(rec_destination=destination,
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
    settings["save_location"] = this.trigger.rec_destination
    if settings["device"] == -1:
        this.trigger.recording_device = 1
        #TODO: Ändern für automatische Auswahl des iMic-Microphone
    else:
        this.trigger.recording_device = settings["device"]
    settings["status"] = "ready"
    settings["status"] = "ready"
    logger.debug("Emitting changed settings to server...")
    client.emit("settingsChanged", settings)
    settings = this.trigger.settings


@client.on("changeStatus")
def on_status_update(action: dict) -> None:
    """Function to handle status updates. Receives an action dictionary containing actions for the recorder and trigger.
    Possible actions are: start [trigger & recorder], stop [trigger & recorder], reset [trigger].

    Parameters
    ----------
    action : dict
        Dictionary containing the action trigger.
    """
    logger.info(f"Change status event received: {action}")
    if action["trigger"] == "start":
        # only do smth if trigger is not already running
        if not this.trigger.stream_thread_is_running:
            logger.info(f"Starting trigger, device: {this.trigger.recording_device}")
            this.trigger.start_trigger()
            client.emit("statusChanged", {"status": "running", "save_location": this.trigger.rec_destination})
    if action["trigger"] == "stop":
        # only do smth if trigger is currently running
        if this.trigger.stream_thread_is_running:
            this.trigger.stop_trigger()
            logger.info("Trigger stopped.")
            client.emit("statusChanged", {"status": "ready", "save_location": this.trigger.rec_destination})
    if action["trigger"] == "reset":
        if not this.trigger.stream_thread_is_running:
            new_rec_destination = this.trigger.voice_field.reset_grid()
            logger.info("Trigger reset.")
            client.emit("statusChanged", {"status": "reset", "save_location": new_rec_destination})


@client.on("removeRecording")
def on_remove_recording(grid_location: dict) -> None:
    """Function to handle the "removeRecording" event. Removes the recording at the specified grid location.

    Parameters
    ----------
    grid_location : dict
        Dictionary containing the grid location of the recording to be removed. {freqBin, dbBin}
    """
    this.trigger.voice_field.grid[grid_location["dbaBin"]][grid_location["freqBin"]] = None
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