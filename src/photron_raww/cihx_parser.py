from typing import List, Tuple
from xml.etree.ElementTree import fromstring


def read_file(path: str) -> List[str]:
    """Reads the content of a file and returns it as a list of strings.

    Parameters
    ----------
    path : str
        The path to the file to be read.

    Returns
    -------
    List[str]
        A list containing the lines of the file as strings.
    """
    with open(path, errors="ignore") as file:
        content = file.readlines()
    return content


def read_file_as_xml(path: str) -> str:
    """Reads the content of a CIHX file, extracts a specific XML section identified by the "cih" tag,
    and returns it as an XML string.

    Parameters
    ----------
    path : str
        The path to the file to be read.

    Returns
    -------
    str
        An XML string containing the content within the "cih" tag.
    """
    content = read_file(path)
    # transform cih tag to xml string
    opening, closing = get_tag_line_idx(content, "cih")[0]
    xml = ['<?xml version="1.0" encoding="utf-8"?>\n']
    xml.extend(content[opening:closing + 1])
    xml = "".join(xml)
    return xml


def get_tag_line_idx(lines: List[str], tag: str) -> List[Tuple[int, int]]:
    """Returns a list of tuples containing opening and closing line indices for the provided tag.
    The tag should be specified by its name (e.g., 'html') and not including '<' or '</>'.

    Parameters
    ----------
    lines : List[str]
        List of lines inside the file.
    tag : str
        The tag name for which to find opening and closing indices.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples, each containing opening and closing line indices for the specified tag.
    """
    opening_line = -1
    tag_idx_list = []
    for idx, line in enumerate(lines):
        if f"<{tag}>" in line:
            opening_line = idx
        if f"</{tag}>" in line:
            closing_line = idx
            tag_idx_list.append((opening_line, closing_line))
    return tag_idx_list


def get_image_data_info(path: str):
    """Parses a CIHX file containing image data information and returns a dictionary with relevant data.

    Note
    ----
    Returned dictionary contains keys:
        - 'width': Width of the image.
        - 'height': Height of the image.
        - 'colorInfo': Bit information about the image's color.

    Parameters
    ----------
    path : str
        The path to the file to be read.

    Returns
    -------
    dict
        Containing image data information with keys 'width', 'height' and 'colorInfo'.
    """
    xml = read_file_as_xml(path)
    tree = fromstring(xml)
    image_data_dict = {
        "width": int(tree.find("./imageDataInfo/resolution/width").text),
        "height": int(tree.find("./imageDataInfo/resolution/height").text),
        "colorInfo": int(tree.find("./imageDataInfo/colorInfo/bit").text),
    }
    return image_data_dict
