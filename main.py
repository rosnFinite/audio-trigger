import logging
import multiprocessing
import time
import os

from backend.server import run_server
from backend.client import run_client
from file_watcher.watcher import run_watcher
from config_utils import CONFIG

logging.basicConfig(
    format='%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    processes = []
    try:
        proc = multiprocessing.Process(target=run_watcher,
                                       args=(CONFIG["camera_recordings_path"], CONFIG["client_recordings_path"], ))
        processes.append(proc)
        proc.start()

        proc = multiprocessing.Process(target=run_server)
        processes.append(proc)
        proc.start()

        proc = multiprocessing.Process(target=run_client)
        processes.append(proc)
        proc.start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) to gracefully terminate processes
        for proc in processes:
            proc.terminate()
    except Exception as e:
        logger.exception(e)

