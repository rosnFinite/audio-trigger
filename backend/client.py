import sys
import socketio
import logging

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
def on_connect():
    print("connected to server")


@client.on("changeSettings")
def on_settings_change(settings):
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
    settings["status"]["recorder"] = "ready"
    settings["status"]["trigger"] = "ready"
    logging.info("Emitting changes in recorder/trigger status...")
    client.emit("settingsChanged", settings)


@client.on("changeStatus")
def on_status_update(action):
    if action["trigger"] == "start":
        # only do smth if trigger is not already running
        if not this.trigger.stream_thread_is_running:
            this.trigger.start_trigger(1)
            client.emit("statusChanged", {"recorder": "running", "trigger": "running"})
    if action["trigger"] == "stop":
        # only do smth if trigger is currently running
        if this.trigger.stream_thread_is_running:
            this.trigger.stop_trigger()
            client.emit("statusChanged", {"recorder": "ready", "trigger": "ready"})
    if action["trigger"] == "reset":
        if not this.trigger.stream_thread_is_running:
            this.trigger.grid.reset_grid()
            client.emit("statusChanged", {"recorder": "ready", "trigger": "ready"})


@client.on("startTrigger")
def on_start_trigger(device_idx):
    if not this.trigger.stream_thread_is_running:
        this.trigger.start_trigger(device_idx)
    client.emit("statusChanged", {"recorder": "running", "trigger": "running"})


if __name__ == "__main__":
    client.connect("http://localhost:5001")
    client.wait()
