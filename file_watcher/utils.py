import logging
import time
import threading

import matplotlib.pyplot as plt
import numpy as np

plot_lock = threading.Lock()

def create_visualizations(get_event):
    start = time.time()
    logging.debug(f"Creating visualizations for {get_event['dir_path']}, Identifier: {get_event['id']}...")
    logging.debug(f"Plotting waveform for {get_event['dir_path']}...")

    sound = get_event["parsel_sound"]
    parent_dir = get_event["dir_path"]

    with plot_lock:
        plot_waveform(sound, parent_dir)
        plot_spectrogram_and_intensity(sound, parent_dir)
    logging.debug(f"TIME: {time.time() - start}")


def plot_waveform(data, location):
    plt.figure()
    plt.plot(data.xs(), data.values.T)
    plt.xlim([data.xmin, data.xmax])
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.savefig(f"{location}/waveform.png")
    logging.debug(f"Waveform plot saved to {location}\\waveform.png")
    plt.close()


def plot_spectrogram_and_intensity(data, location):
    plt.figure()

    spectogram = data.to_spectrogram()
    X, Y = spectogram.x_grid(), spectogram.y_grid()
    sg_db = 10 * np.log10(spectogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 70, cmap='afmhot')
    plt.ylim([spectogram.ymin, spectogram.ymax])
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frequenz [Hz]")

    plt.twinx()

    intensity = data.to_intensity()
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("Intensität [dB]")

    plt.xlim([data.xmin, data.xmax])
    plt.savefig(f"{location}/spectrogram_intensity.png")
    logging.debug(f"Spectrogram and intensity plot saved to {location}\\spectrogram_intensity.png")
    plt.close()
