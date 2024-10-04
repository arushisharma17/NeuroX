# utils.py

import json

def load_config(config_file):
    """
    Loads configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration data.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

