import yaml
import os
import logging

logger = logging.getLogger(__name__)


def get_config() -> dict:
    """Function to load the config file and return the config dictionary.
    If the config file has already been loaded, the function returns the cached config dictionary.

    Returns
    -------
    dict
        Dictionary containing the configuration settings.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, ".."))
    path_to_config = os.path.join(project_dir, "config.yaml")
    if not hasattr(get_config, "config"):
        logger.info(f"Loading config from {path_to_config}")
        with open(path_to_config) as file:
            get_config.config = yaml.load(file, Loader=yaml.FullLoader)
    return get_config.config


CONFIG = get_config()
