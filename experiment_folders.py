import os
from datetime import datetime
from pathlib import Path

import yaml


def setup_experiment(config_file_path):
    """
    Sets up experiment folders and returns configuration from the given file path.
    1. Create a directory under `experiments/{experiment_name}/{timestamp}`. Experiment name will be the name of the
    configuration file.
    2. Load configuration from the given YAML file and dump its contents to experiment
    path
    3. Return configuration as a dictionary along with the experiment path
    :param config_file_path: relative or absolute path to configuration yaml file
    :return: configuration as a dictionary

        Example:
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdirname:
    ...     config_path = Path(tmpdirname) / "test_config.yml"
    ...     with open(config_path, "w") as f:
    ...         yaml.safe_dump({"param1": "value1", "param2": "value2"}, f)
    ...     config = setup_experiment(config_path)  # doctest: +ELLIPSIS
    ...     assert config["param1"] == "value1"
    ...     assert config["param2"] == "value2"
    ...     assert "output_dir" in config
    ...     assert os.path.exists(config["output_dir"])
    ...     assert os.path.exists(Path(config["output_dir"]) / "config.yml")
    Setting up experiment experiments/test_config/...
    """
    config_file_path = os.path.abspath(config_file_path)

    filename = Path(os.path.splitext(config_file_path)[0]).name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = os.path.join("experiments", filename, timestamp)
    print(f"Setting up experiment {experiment_path}")

    # Open and load yaml file
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file {config_file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"Configuration file {config_file_path} not found!")
        return None

    # Create experiment directory to write logs and config file
    os.makedirs(experiment_path, exist_ok=True)

    # Add experiment path to config
    config["output_dir"] = experiment_path

    return config

def dump_config(config):
    experiment_path = config["output_dir"]
    # Dump yaml content to config file under experiment directory
    config_dump_path = os.path.join(experiment_path, "config.yml")
    try:
        with open(config_dump_path, "w") as file:
            yaml.safe_dump(config, file)
    except Exception as e:
        print(f"Error writing file {config_dump_path}: {e}")
        return None