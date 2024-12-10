#!/usr/bin/env python3

import socket
from datetime import datetime
from pathlib import Path
import logging

import yaml
from ..utils import add_default_repr

from abc import ABC, abstractmethod
from ..utils import setup_logger, setup_default_logger


# noinspection PyShadowingNames
@add_default_repr
class Experiment(ABC):
    """
    Class for running experiments.

    Trains a model to relate neural and stimuli data. The model can be a linear model, a multilayer perceptron, or a
    gradient boosted tree. The experiment can be a decoding experiment. The model uses cross-validation to train and predict on left-out folds.
    The parameters, predictions, and ground truth are saved to a results file.

    This class is responsible for:
    - loading the data
    - running the experiment
    - saving the results

    Parameters:
        config (dict): dictionary of configuration parameters

    Attributes:
        config (dict): dictionary of configuration parameters
        video_data_loader (VideoDataLoader): dictionary of video data
        neural_data_loader (NeuralDataLoader): dictionary of neural data
        experiment_name (str): string of the experiment name
        save_dir (Path): path of the save directory

    Methods:
        get_video_variable: returns the video variable
        get_neural_variable: returns the neural variable
        block_frames: splits the movie into blocks of frames
        create_save_dir: creates the save directory
        save_results: saves the results
        run_experiment: runs the experiment

    Usage:

    ```python
    from experiments import ExperimentRunner
    from neural_data_analysis import recursive_dict_update
    from datetime import date
    import logging
    from pathlib import Path
    ```

    """

    def __init__(self, config: dict, load_only: bool = False):
        """
        Initialize the ExperimentRunner class with the configuration parameters.

        Parameters:
            config (dict): dictionary of configuration parameters, typically saved and loaded as a yaml file.
            load_only (bool): Whether to initialize the class without running the experiment.
                If false, will create a save directory and output the config file and log file to the directory.

        """
        self.config: dict = config
        self.load_only = load_only

        if load_only:
            self.logger = setup_default_logger()
            self.logger.info(
                "LOAD ONLY MODE for ExperimentRunner class. No experiment_name or save_dir created."
            )
        else:
            # This function gets overloaded by the child class implementation so it will jump right to the child._initialize_experiment()
            self._initialize_experiment()

        # Placeholder for data
        self.neural_data: dict = None
        self.video_data: dict = None

    def _initialize_experiment(self):
        # Create experiment name and save directory
        self.experiment_name = self._create_experiment_name()
        self.save_dir = self._create_save_dir()

    def _create_experiment_name(self) -> str:
        """
        Returns:
            str: string of the experiment name
        """
        date = datetime.now().strftime("%Y-%m-%d")
        project_name = self.config.get("project_name", "project")
        full_experiment_name = f"{date}_{project_name}"
        return full_experiment_name

    def _create_save_dir(self) -> Path:
        """
        Create a save directory for the results.

        The directory name is made up of the experiment name and an index.

        Returns:
            save_dir (Path): Path to the save directory.
        """
        experiment_name = self.experiment_name

        # Get the full hostname
        full_hostname = socket.gethostname()
        # Split the hostname at '.' and take the first part
        short_hostname = full_hostname.split("-")[0]

        results_id = 1
        save_dir = Path(f"results/{experiment_name}_{short_hostname}_{results_id}")

        while Path(save_dir).exists():
            results_id += 1
            save_dir = Path(f"results/{experiment_name}_{short_hostname}_{results_id}")

        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir

    def _save_config(self, config):
        config_filename = f"{self.experiment_name}_config.yaml"
        with open(self.save_dir / config_filename, "w") as f:
            yaml.dump(config, f)
        self.logger.info(
            f"SAVED: Config file [{config_filename}] to [{self.save_dir}].\n"
        )

    @abstractmethod
    def _load_data(self):
        """Load the data required for the experiment."""
        pass

    @abstractmethod
    def _save_results(self):
        """Save the results of the experiment."""
        pass

    def run_experiment(self):
        """Run the experiment workflow."""
        self.logger.info("Starting experiment...")
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()
        self.save_results()
        self.logger.info("Experiment completed.")


if __name__ == "__main__":
    pass
