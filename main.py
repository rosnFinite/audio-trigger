import logging
import multiprocessing
import time

from backend.server import run_server
from backend.client import run_client
from file_watcher.file_watcher import run_file_watcher

if __name__ == "__main__":
    processes = []
    try:
        proc = multiprocessing.Process(target=run_file_watcher,
                                       args=("C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\backend\\recordings",))
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
        logging.exception(e)

