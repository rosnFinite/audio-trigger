import random
import time

import socketio

sio = socketio.Client()
sio.connect("http://localhost:5001")


def main():
    while True:
        time.sleep(1)
        data = {"x": random.randint(0, 35), "y": random.randint(0, 15)}
        print(f"sending {data}")
        sio.emit("data", data)


if __name__ == "__main__":
    main()
