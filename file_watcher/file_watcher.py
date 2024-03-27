import concurrent.futures
import logging
import time
from queue import Queue
from threading import Thread
import random

import parselmouth
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from utils import create_visualizations

logging.basicConfig(
    format='%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class ClientRecordingsFileHandler(PatternMatchingEventHandler):
    def __init__(self, queue):
        PatternMatchingEventHandler.__init__(self, patterns=["*.wav"])
        self.queue = queue

    def on_created(self, event):
        identifier = random.randint(0, 10000)
        logging.info(f"Identifier: {identifier} File created: {event.src_path}")
        parent_dir = "\\".join(event.src_path.split("\\")[:-1])
        # solution to fix issue of file not being fully created yet
        # on creation event does not take writing process into account
        snd = None
        while snd is None:
            try:
                snd = parselmouth.Sound(event.src_path)
            except (parselmouth.PraatError, TypeError):
                snd = None
                logging.warning(f"Not yet finished creating {event.src_path}...")
                time.sleep(0.1)
        self.queue.put({"id": identifier, "dir_path": parent_dir, "parsel_sound": snd})


def start_watchdog(watchdog_queue, dir_path):
    logging.info("Starting watchdog observer...")
    event_handler = ClientRecordingsFileHandler(queue=watchdog_queue)

    observer = Observer()
    observer.schedule(event_handler, path=dir_path, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        observer.stop()
        logging.error(f"An error occurred: {e}")
    finally:
        observer.join()
        logging.info("Watchdog observer stopped.")


if __name__ == "__main__":
    dir_path = "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\backend\\recordings"

    watchdog_queue = Queue()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    logging.info("Starting watchdog observer thread...")
    worker = Thread(target=start_watchdog, name="Watchdog", args=(watchdog_queue, dir_path), daemon=True)
    worker.start()

    while True:
        if not watchdog_queue.empty():
            pool.submit(create_visualizations, watchdog_queue.get())
        else:
            time.sleep(1)
