import random
import time
from threading import Thread

import socketio

from recorder import Trigger, AudioRecorder

"""
sio = socketio.Client()
sio.connect("http://localhost:5001")


def main():
    while True:
        time.sleep(1)
        data = {"x": random.randint(0, 35), "y": random.randint(0, 15)}
        print(f"sending {data}")
        sio.emit("data", data)
"""

if __name__ == "__main__":
    client = socketio.Client()
    client.connect("http://localhost:5001")

    # Probleme bei Treiber von bspw. Razer Headset
    trigger = Trigger(buffer_size=0.2, rec_destination="DUMMY", socket=client)
    trigger.start_trigger(input_device_index=1)

