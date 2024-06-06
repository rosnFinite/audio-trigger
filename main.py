import logging
import multiprocessing
import sys
import time

from src.backend.server import run_server
from src.backend.client import run_client
from src.file_watcher.watcher import run_watcher
from src.config_utils import CONFIG

# solution for path problems using vscode
sys.path.append("D:\\rosef\\audio-trigger")

logging.basicConfig(
    format='%(levelname)-8s | %(asctime)s | %(filename)s%(lineno)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def gracefully_terminate(processes):
    for proc in processes:
        proc.terminate()

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
        gracefully_terminate(processes=processes)
    except FileNotFoundError as e:
        logger.critical(f"Festgelegter Speicherort für den Triggerprozess [{CONFIG['client_recordings_path']}] oder für die Kamera [{CONFIG['camera_recordings_path']}] konnte nicht gefunden werden.")
        # gracefully terminating processes
        gracefully_terminate(processes=processes)
    except Exception as e:
        logger.exception(e)

