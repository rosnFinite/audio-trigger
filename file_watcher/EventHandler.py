import logging
import os
import random
import time
from queue import Queue
from typing import List

import parselmouth
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

from config_utils import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ClientRecordingsFileHandler(PatternMatchingEventHandler):
    def __init__(self, queue: Queue):
        PatternMatchingEventHandler.__init__(self, patterns=["*.wav"])
        self.queue = queue

    def on_created(self, event: FileSystemEvent) -> None:
        """Event handler for file creation events. Will create a new parselmouth.Sound object from the file and store it
        in the queue for further processing.

        Parameters
        ----------
        event : FileSystemEvent
            The event object containing information about the file creation event.
        """
        identifier = random.randint(0, 10000)
        logger.info(f"Identifier: {identifier} File created: {event.src_path}")
        parent_dir = os.path.split(event.src_path)[0]
        # solution to fix issue of file not being fully created yet
        # on creation event does not take writing process into account
        snd = None
        while snd is None:
            try:
                snd = parselmouth.Sound(event.src_path)
            except (parselmouth.PraatError, TypeError):
                snd = None
                logger.warning(f"Not yet finished creating {event.src_path}...")
                time.sleep(0.1)
        self.queue.put({"id": identifier, "dir_path": parent_dir, "parsel_sound": snd})


class CameraRecordingsFileHandler(PatternMatchingEventHandler):
    def __init__(self, queue: Queue):
        PatternMatchingEventHandler.__init__(self, patterns=["*.cihx"])
        self.queue = queue

    def on_created(self, event: FileSystemEvent) -> None:
        """Event handler for file creation events.

        Parameters
        ----------
        event : FileSystemEvent
            The event object containing information about the file creation event.
        """
        identifier = random.randint(0, 10000)
        logger.info("Camera recording finished. CIHX File was created: %s", event.src_path)
        logger.info("Extracting 4 images and converting them to JPEG...")
        parent_dir = os.path.split(event.src_path)[0]
        # extract 4 images from the folder in which the cihx file was created
        # Fetch all image files
        image_files = [f for f in os.listdir(parent_dir) if
                       f.lower().endswith('.raww')]
        total_frames = len(image_files)
        if total_frames < 4:
            indices = list(range(total_frames))
        else:
            indices = [
                0,
                total_frames - 1,
                total_frames // 3,
                2 * total_frames // 3
            ]
        selected_files = [os.path.join(parent_dir, image_files[i]) for i in indices]
        # find the last created audio recording folder, information is stored in the .latest_trigger filed in
        # the client recordings folder
        client_recordings_path = CONFIG["client_recordings_path"]
        latest_trigger_file = os.path.join(client_recordings_path, ".latest_trigger")
        with open(latest_trigger_file, "r") as f:
            last_trigger = f.read().strip()

        # put the selected files into the queue for further processing
        self.queue.put({"id": identifier,
                        "dir_path": parent_dir,
                        "meta": event.src_path,
                        "images": selected_files,
                        "dir_trigger": last_trigger,
                        })


def start_watchdog(plotting_q: Queue, image_q: Queue, paths_to_watch: List[str]) -> None:
    """Starts the watchdog observer to monitor the specified directories for new files. Will monitor all subdirectories.

    Parameters
    ----------
    plotting_q : Queue
        parselmouth.Sound objects to be processed into visualizations.
    image_q : Queue
        paths of .raww image files to be transformed into jpg images.
    paths_to_watch : list of str
        The paths to the directories to monitor. [camera_recordings_path, client_recordings_path]
    """
    camera_recordings_path, client_recordings_path = paths_to_watch
    logger.info("Starting watchdog observer...")
    event_handler = ClientRecordingsFileHandler(queue=plotting_q)
    camera_event_handler = CameraRecordingsFileHandler(queue=image_q)

    observer = Observer()
    if client_recordings_path is not None:
        observer.schedule(event_handler, path=client_recordings_path, recursive=True)
    if camera_recordings_path is not None:
        observer.schedule(camera_event_handler, path=camera_recordings_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        observer.stop()
        logger.error(f"An error occurred: {e}")
    finally:
        observer.join()
        logger.info("Watchdog observer stopped.")
