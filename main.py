import logging
import multiprocessing
import sys
import time

from backend.server import run_server
from backend.client import run_client

if __name__ == "__main__":
    try:
        processes = []

        proc = multiprocessing.Process(target=run_server)
        processes.append(proc)
        proc.start()

        proc = multiprocessing.Process(target=run_client)
        processes.append(proc)
        proc.start()

        while True:
            time.sleep(1)
    except Exception as e:
        logging.exception(e)

