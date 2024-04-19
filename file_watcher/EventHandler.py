import logging
import os
import random
import time
from queue import Queue

import parselmouth
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent
from watchdog.observers import Observer

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


def start_watchdog(q: Queue, path_to_watch: str) -> None:
    """Starts the watchdog observer to monitor the specified directory for new files. Will monitor all subdirectories.

    Parameters
    ----------
    q : Queue
        The queue to store the events.
    path_to_watch : str
        The path to the directory to monitor.
    """
    logger.info("Starting watchdog observer...")
    event_handler = ClientRecordingsFileHandler(queue=q)

    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=True)
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
