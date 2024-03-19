import sys
import socketio
import logging

from typing import List, Dict

from recorder import Trigger

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

client = socketio.Client(logger=True, engineio_logger=True)
this = sys.modules[__name__]
this.trigger = None


@client.on("connect")
def on_connect() -> None:
    """This function is called when the client successfully connects to the server.
    It prints a message indicating that the connection has been established.
    """
    print("connected to server")
    print("Registering as audio trigger client...")
    client.emit("registerClient", {"type": "audio"})


@client.on("clients")
def on_clients(clients: list) -> None:
    """Event handler for the "clients" event.

    Parameters
    ----------
    clients: List[Dict[str]]
        The data received from the audio trigger client. Contains the client's session ID.
    """
    print("received clients... ", clients)


@client.on("changeSettings")
def on_settings_change(settings: dict) -> None:
    """Function to handle settings change event.

    Parameters
    ----------
    settings : dict
        Dictionary containing the updated settings.
    """
    logging.info("Creating new trigger instance..")
    this.trigger = Trigger(min_q_score=settings["qualityScore"],
                           semitone_bin_size=settings["frequency"]["steps"],
                           freq_bounds=(settings["frequency"]["lower"], settings["frequency"]["upper"]),
                           dba_bin_size=settings["db"]["steps"],
                           dba_bounds=(settings["db"]["lower"], settings["db"]["upper"]),
                           buffer_size=settings["bufferSize"],
                           channels=1 if settings["mono"] else 2,
                           rate=settings["sampleRate"],
                           chunksize=settings["chunkSize"],
                           socket=client)
    if settings["device"] == -1:
        this.trigger.recording_device = 1
        #TODO: Ändern für automatische Auswahl des iMic-Microphone
    else:
        this.trigger.recording_device = settings["device"]
    settings["status"]["recorder"] = "ready"
    settings["status"]["trigger"] = "ready"
    logging.info("Emitting changes in recorder/trigger status...")
    client.emit("settingsChanged", settings)


@client.on("changeStatus")
def on_status_update(action: dict) -> None:
    """Function to handle status updates. Receives an action dictionary containing actions for the recorder and trigger.
    Possible actions are: start [trigger & recorder], stop [trigger & recorder], reset [trigger].

    Parameters
    ----------
    action : dict
        Dictionary containing the action trigger.
    """
    if action["trigger"] == "start":
        # only do smth if trigger is not already running
        if not this.trigger.stream_thread_is_running:
            print(f"Starting trigger, device {this.trigger.recording_device}")
            this.trigger.start_trigger()
            client.emit("statusChanged", {"recorder": "running", "trigger": "running"})
    if action["trigger"] == "stop":
        # only do smth if trigger is currently running
        if this.trigger.stream_thread_is_running:
            this.trigger.stop_trigger()
            client.emit("statusChanged", {"recorder": "ready", "trigger": "ready"})
    if action["trigger"] == "reset":
        if not this.trigger.stream_thread_is_running:
            this.trigger.grid.reset_grid()
            client.emit("statusChanged", {"recorder": "reset", "trigger": "reset"})


@client.on("removeRecording")
def on_remove_recording(grid_location: dict) -> None:
    """Function to handle the "removeRecording" event. Removes the recording at the specified grid location.

    Parameters
    ----------
    grid_location : dict
        Dictionary containing the grid location of the recording to be removed. {freqBin, dbBin}
    """
    this.trigger.grid.grid[grid_location["dbaBin"]][grid_location["freqBin"]] = None


if __name__ == "__main__":
    try:
        client.connect("http://localhost:5001")
        client.wait()
    except KeyboardInterrupt:
        print("Exiting...")
        client.disconnect()
        sys.exit(0)

# TODO: Bundle into executable to run server an client in seperate processes