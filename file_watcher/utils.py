import logging
import os.path
import time
import threading
from typing import List
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from photron_raww.raww_visualizer import transform_image

plot_lock = threading.Lock()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_visualizations(get_event):
    start = time.time()
    logger.info(f"Creating visualizations for {get_event['dir_path']}, Identifier: {get_event['id']} ...")
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


def create_image_grid(get_event):
    logger.info(f"Creating image grid for {get_event['dir_path']}, Identifier: {get_event['id']}...")
    img_paths: List[str] = get_event["images"]
    meta_path = get_event["meta"]
    # transforming selected images to JPEG and storing them in corresponding recording directory
    raww_to_jpg(img_paths, meta_path, get_event["dir_trigger"])
    # create image grid for frontend visualization
    # get file names for transformed images
    image_files = image_files = [f"{os.path.join(get_event['dir_trigger'], os.path.split(f)[-1].split('.')[0])}.jpg"
                                 for f in img_paths]
    images = [Image.open(f) for f in image_files]
    # assuming all images are the same size
    width, height = images[0].size
    grid = Image.new('RGB', (width * 2, height * 2), 'white')
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (width, 0))
    grid.paste(images[2], (0, height))
    grid.paste(images[3], (width, height))
    grid.save(f"{get_event['dir_trigger']}/image_grid.jpg")


def raww_to_jpg(img_paths: List[str], meta_path: str, save_path: str):
    logger.info(f"Transforming raww images {img_paths} to JPEG...")
    for raww_img in img_paths:
        logger.info(f"Transforming image {raww_img} to JPEG...")
        try:
            transform_image(path=raww_img, save_path=save_path, metadata_path=meta_path)
        except ValueError as e:
            logger.error(e)
            continue

