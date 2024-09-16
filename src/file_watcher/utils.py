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

from src.photron_raww.raww_visualizer import transform_image

plot_lock = threading.Lock()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_visualizations(get_event):
    """
    Create visualizations for the provided event.

    Parameters
    ----------
    get_event : dict
        Dictionary containing information about the event.
    """
    logger.info(f"Creating visualizations for {get_event['dir_path']}, Identifier: {get_event['id']} ...")
    logger.info(f"Plotting waveform for {get_event['dir_path']}...")

    sound = get_event["parsel_sound"]
    parent_dir = get_event["dir_path"]

    intensity = sound.to_intensity()
    spectrogram = sound.to_spectrogram()
    pitch = sound.to_pitch()

    # get daq measurements
    daq_data = None
    daq_header = None
    if os.path.exists(os.path.join(parent_dir, "measurements.csv")):
        daq_csv = np.genfromtxt(os.path.join(parent_dir, "measurements.csv"), delimiter=",")
        daq_header = daq_csv[0]
        daq_data = daq_csv[1:]
    
    egg_path = os.path.join(parent_dir, "egg.npy")
    egg_data = None
    if os.path.exists(egg_path):
        egg_data = np.load(egg_path)

    with plot_lock:
        plot_waveform(sound, parent_dir)
        plot_spectrogram_and_intensity(sound, spectrogram, intensity, parent_dir)
        if egg_data is not None:
            plot_egg_data(egg_data, parent_dir)
        if daq_data is not None:
            plot_daq_data(daq_data, daq_header, parent_dir)
        # store parselmouth pitch information
        logger.debug(f"Saving parselmouth stats to {parent_dir}...")
        with open(os.path.join(parent_dir, "parsel_stats.txt"), "w") as f:
            print(sound, file=f)
            print(pitch, file=f)
            print(intensity, file=f)


def plot_waveform(data, location):
    """
    Plot the waveform of the provided data object.

    Parameters
    ----------
    data : Parselmouth.Sound
        Sound object.
    location : str
        The directory where the plot will be saved.
    """
    plt.figure()
    plt.plot(data.xs(), data.values.T)
    plt.xlim([data.xmin, data.xmax])
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.savefig(f"{location}/waveform.png")
    logger.info(f"Waveform plot saved to {location}\\waveform.png")
    plt.close()


def plot_spectrogram_and_intensity(sound, spectrogram, intensity, location):
    """
    Plot the spectrogram and intensity of the provided sound object.

    Parameters
    ----------
    sound : Parselmouth.Sound
        Sound object.
    spectrogram : Parselmouth.Spectrogram
        Spectrogram object.
    intensity : Parselmouth.Intensity
        Intensity object.
    location: str
        The directory where the plot will be saved.
    """
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
    

def plot_egg_data(egg, location):
    """
    Creates a plot of the EGG data.
    
    Parameters
    ----------
    egg: np.ndarray
        EGG data to be plotted.
    location: str
        The directory where the plot will be saved.
    """
    if egg is None:
        return
    plt.figure()
    
    plt.plot(egg)
    plt.savefig(f"{location}/egg.png")
    plt.close()


def plot_daq_data(data, header, location):
    """
    Creates a plot of the DAQ data.

    Parameters
    ----------
    data: np.ndarray
        DAQ data to be plotted.
    header: np.ndarray
        Header information for the DAQ data.
    location: str
        The directory where the plot will be saved.
    """
    time_col = data[:, 0]
    for i, channel in enumerate(header, start=1):
        plt.figure()
        plt.plot(time_col, data[:, i], label=channel)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.grid(True)
        plt.savefig(f"{location}/daq_data_{channel}.png")
        logger.info(f"DAQ measurement plot for {channel} saved to {location}\\daq_data_{channel}.png")
        plt.close()

def create_image_grid(get_event):
    """
    Create an image grid for the provided event.

    Parameters
    ----------
    get_event : dict
        Dictionary containing information about the event.
    """
    logger.info(f"Creating image grid for {get_event['dir_path']}, Identifier: {get_event['id']}...")
    img_paths: List[str] = get_event["images"]
    meta_path = get_event["meta"]
    # transforming selected images to JPEG and storing them in corresponding recording directory
    raww_to_jpg(img_paths, meta_path, get_event["dir_trigger"])
    # create image grid for frontend visualization
    # get file names for transformed images
    image_files = image_files = [f"{os.path.join(get_event['dir_trigger'], os.path.split(f)[-1].split('.')[0])}.png"
                                 for f in img_paths]
    images = [Image.open(f) for f in image_files]
    # assuming all images are the same size
    width, height = images[0].size
    grid = Image.new('RGB', (width * 2, height * 2), 'white')
    grid.paste(images[0], (0, 0))
    grid.paste(images[1], (width, 0))
    grid.paste(images[2], (0, height))
    grid.paste(images[3], (width, height))
    grid.save(f"{get_event['dir_trigger']}/image_grid.png")


def raww_to_jpg(img_paths: List[str], meta_path: str, save_path: str):
    """
    Transform raww images provided as a list of paths to JPEG and stores them in the specified directory.

    Parameters
    ----------
    img_paths : List[str]
        List of paths to raww images to be transformed.
    meta_path : str
        Path to the metadata file with a '.cihx' extension.
    save_path : str
        The directory where the transformed images will be saved.
    """
    logger.info(f"Transforming raww images {img_paths} to JPEG...")
    for raww_img in img_paths:
        logger.info(f"Transforming image {raww_img} to JPEG...")
        try:
            transform_image(path=raww_img, save_path=save_path, metadata_path=meta_path)
        except ValueError as e:
            logger.error(e)
            continue

if __name__ == "__main__":
    event = {
        "id":"111",
        "dir_path": "anything",
        "dir_trigger": "C:\\Users\\fabio\PycharmProjects\\audio-trigger\\src\\photron_raww\\test\\data",
        "images": [
            "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\src\photron_raww\\test\\data\\TEST_C001H001S0002000001.raww",
            "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\src\photron_raww\\test\\data\\TEST_C001H001S0002000001.raww",
            "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\src\photron_raww\\test\\data\\TEST_C001H001S0002000001.raww",
            "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\src\photron_raww\\test\\data\\TEST_C001H001S0002000001.raww"
        ],
        "meta": "C:\\Users\\fabio\\PycharmProjects\\audio-trigger\\src\\photron_raww\\test\\data\\TEST_C001H001S0002.cihx"
    }

    create_image_grid(event)