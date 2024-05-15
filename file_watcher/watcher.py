import concurrent.futures
import logging
import time
from queue import Queue
from threading import Thread

from .EventHandler import start_watchdog
from .utils import create_visualizations, create_image_grid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_watcher(camera_recordings_path: str = None, client_recordings_path: str = None):
    """Function to start the file watcher thread. Will start the watchdog observer thread and create a thread pool to
    handle the visualization creation tasks.

    Parameters
    ----------
    camera_recordings_path : str
        The path to the parent directory of the camera recordings.
    client_recordings_path : str
        The path to the parent directory of the client recordings.
    """
    plotting_queue = Queue()
    camera_image_queue = Queue()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    logger.info("Starting watchdog observer thread...")
    worker = Thread(target=start_watchdog,
                    name="Watchdog",
                    args=(plotting_queue, camera_image_queue, [camera_recordings_path, client_recordings_path]), daemon=True)
    worker.start()

    while True:
        if not plotting_queue.empty():
            pool.submit(create_visualizations, plotting_queue.get())
        if not camera_image_queue.empty():
            pool.submit(create_image_grid, camera_image_queue.get())
        else:
            time.sleep(1)


if __name__ == "__main__":
    dir_path = "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\backend\\recordings"

    run_watcher(dir_path)
