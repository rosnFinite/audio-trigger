import logging
import time
import threading

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plot_lock = threading.Lock()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_visualizations(get_event):
    start = time.time()
    logger.info(f"Creating visualizations for {get_event['dir_path']}, Identifier: {get_event['id']}...")
    logger.info(f"Plotting waveform for {get_event['dir_path']}...")

    sound = get_event["parsel_sound"]
    parent_dir = get_event["dir_path"]

    intensity = sound.to_intensity()
    spectrogram = sound.to_spectrogram()
    pitch = sound.to_pitch()

    with plot_lock:
        plot_waveform(sound, parent_dir)
        plot_spectrogram_and_intensity(sound, spectrogram, intensity, parent_dir)
    # store parselmouth pitch information

    with open(f"{parent_dir}/parsel_stats.txt", "w") as f:
        print(sound, file=f)
        print(pitch, file=f)
        print(intensity, file=f)


def plot_waveform(data, location):
    plt.figure()
    plt.plot(data.xs(), data.values.T)
    plt.xlim([data.xmin, data.xmax])
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.savefig(f"{location}/waveform.png")
    logger.info(f"Waveform plot saved to {location}\\waveform.png")
    plt.close()


def plot_spectrogram_and_intensity(sound, spectrogram, intensity, location):
    plt.figure()

    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - 70, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("Zeit [s]")
    plt.ylabel("Frequenz [Hz]")

    plt.twinx()

    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("Intensität [dB]")

    plt.xlim([sound.xmin, sound.xmax])
    plt.savefig(f"{location}/spectrogram_intensity.png")
    logger.info(f"Spectrogram and intensity plot saved to {location}\\spectrogram_intensity.png")
    plt.close()
