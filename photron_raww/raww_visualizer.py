import os
import numpy as np

from PIL import Image
from typing import Optional

from . import cihx_parser


def transform_images(path: str, save_path: str, metadata_path: Optional[str] = None) -> None:
    """Transforms multiple raw image files (.raww) in a directory using metadata information and saves the results as
    JPEG images.

    Note
    ----
    - The raw image files should have a '.raww' extension.
    - The metadata file should have a '.cihx' extension.
    - Transformed images are saved in JPEG format with filenames derived from the original raw image files.
    - 16-bit color information will be lost in the current version.

    Parameters
    ----------
    path : str
        The path to the directory containing raw image files with a '.raww' extension.

    save_path : str
        The directory where the transformed images will be saved.

    metadata_path : str
        The path to the metadata file with a '.cihx' extension.
        If not provided, the function looks for a '.cihx' file in the input directory.

    Raises
    ------
        ValueError
            If the provided path is not a directory, no metadata file is found, or no '.raww' files are present.

    Examples
    --------
    Transform all images in a given directory with automatically loading available metadata:

    >>> transform_images("test/data", "test/output")

    Transform all images in a given direcotry with manually providing path to corresponding metadata:

    >>> transform_images("test/data", "test/output", "test/data/TEST_C001H001S0002.cihx")
    """
    raww_files = []
    if os.path.isdir(path):
        # find all .raww files to transform
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".raww"):
                    raww_files.append(file)
                if file.endswith(".cihx") and metadata_path is None:
                    metadata_path = f"{path}/{file}"
    else:
        raise ValueError("Provided path is not a directory.")
    if metadata_path is None:
        raise ValueError("Provided directory does not contain metadata file (.cihx) or no "
                         "path to metadata file has been provided.")
    if len(raww_files) == 0:
        raise ValueError("Provided directory does not contain any .raww files to be transformed.")
    img_metadata = cihx_parser.get_image_data_info(metadata_path)
    # iterate over raww files to be transformed
    for raww in raww_files:
        bin_image = np.fromfile(f"{path}/{raww}", dtype="uint16")
        # Pillow.Image transforms 16-bit color information to 8-bit
        img = Image.fromarray(bin_image.reshape(img_metadata["width"], img_metadata["height"])).convert("L")
        filename = raww.split(".")[0]
        img.save(f"{save_path}/{filename}.jpg")


def transform_image(path: str, save_path: str, metadata_path: str):
    """Transforms a raw image file (.raww) using metadata information and saves the result as a JPEG image.

    Note
    ----
    - The raw image file should have a '.raww' extension.
    - The metadata file should have a '.cihx' extension.
    - The transformed image is saved in JPEG format with a filename derived from the original raw image file.
    - 16-bit color information will be lost in the current version.

    Parameters
    ----------
    path : str
        The path to the raw image file with a '.raww' extension.
    save_path : str
        The directory where the transformed image will be saved.
    metadata_path : str
        The path to the metadata file with a '.cihx' extension.

    Raises
    ------
        ValueError
            If the provided image file path or metadata path does not have the expected extensions.

    Examples
    --------
    Transform an image to JPEG format and save it inside save_path

    >>> transform_image("test/data/TEST_C001H001S0002000001.raww", "test/output", "test/data/TEST_C001H001S0002.cihx")
    """
    if not path.endswith(".raww"):
        raise ValueError("Provided image file path does not have .raww extension.")
    if not metadata_path.endswith(".cihx"):
        raise ValueError("Provided path to metadata does not have .cihx extension.")
    img_metadata = cihx_parser.get_image_data_info(metadata_path)
    bin_image = np.fromfile(f"{path}", dtype="uint16")
    # Pillow.Image transforms 16-bit color information to 8-bit
    img = Image.fromarray(bin_image.reshape(img_metadata["width"], img_metadata["height"])).convert("L")
    filename = os.path.split(path)[1].split(".raww")[0]
    img.save(os.path.join(save_path, f"{filename}.jpg"))


if __name__ == "__main__":
    transform_image("test/data/TEST_C001H001S0002000001.raww",
                    metadata_path="test/data/TEST_C001H001S0002.cihx",
                    save_path="test/output")
