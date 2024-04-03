import concurrent.futures
import logging
import time
from queue import Queue
from threading import Thread

from .EventHandler import start_watchdog
from .utils import create_visualizations

logger = logging.getLogger(__name__)


def run_file_watcher(path_to_watch: str):
    """Function to start the file watcher thread. Will start the watchdog observer thread and create a thread pool to
    handle the visualization creation tasks.

    Parameters
    ----------
    path_to_watch : str
        The path to the directory to monitor.
    """
    watchdog_queue = Queue()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    logger.info("Starting watchdog observer thread...")
    worker = Thread(target=start_watchdog, name="Watchdog", args=(watchdog_queue, path_to_watch), daemon=True)
    worker.start()

    while True:
        if not watchdog_queue.empty():
            pool.submit(create_visualizations, watchdog_queue.get())
        else:
            time.sleep(1)


if __name__ == "__main__":
    dir_path = "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\backend\\recordings"

    run_file_watcher(dir_path)
