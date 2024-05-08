import yaml


def get_config() -> dict:
    """Function to load the config file and return the config dictionary.
    If the config file has already been loaded, the function returns the cached config dictionary.

    Returns
    -------
    dict
        Dictionary containing the configuration settings.
    """
    if not hasattr(get_config, "config"):
        with open("config.yaml") as file:
            get_config.config = yaml.load(file, Loader=yaml.FullLoader)
    return get_config.config


CONFIG = get_config()

if __name__ == "__main__":
    print(CONFIG)
