#!/usr/bin/env python3
import itertools
import pickle
import socket
from datetime import datetime
from pathlib import Path
import logging
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from ..utils import add_default_repr
from scipy.special import expit
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
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
        self.config: dict = config.get("ExperimentRunner", {})
        self.load_only = load_only

        # Setup logging and directories
        self._setup_experiment_environment()

        # Log the configuration file and the save directory.
        self.logger.info(
            f"LOADED: Config file [{config['general']['config_file']}] in ExperimentRunner class."
        )
        self.logger.info(f"COMPLETED: Created save directory [{str(self.save_dir)}].")

        # Placeholder for data
        self.neural_data: dict = None
        self.video_data: dict = None

    def _setup_experiment_environment(self):
        if self.load_only:
            self.logger = setup_default_logger()
        else:
            # Create save directory and logger
            self.experiment_name = self.create_experiment_name()
            self.save_dir = self.create_save_dir()
            self.logger = setup_logger(
                logger_name="logger",
                log_filepath=Path(f"{self.save_dir}/{self.experiment_name}.log"),
            )

            # Save the config file
            self._save_config(self.config)

    def create_experiment_name(self) -> str:
        """
        Create experiment name based on the date, project name, experiment name, experiment type, and model name.
        The project name and experiment name are taken from the config file. The experiment type and model name are
        hardcoded in the config file.
        This experiment name is then used for the save directory, log file, and results file.

        Returns:
            full_experiment_name (str): string of the experiment name
        """
        project_name = self.config.get("project_name", "project")
        experiment_name = self.config.get("experiment_name", "experiment")
        date = datetime.now().strftime("%Y-%m-%d")
        full_experiment_name = f"{date}_{project_name}_{self.config['type']}_{self.config['model']}_{experiment_name}"
        return full_experiment_name

    def create_save_dir(self) -> Path:
        """
        Create a save directory for the results.

        The directory name is made up of the experiment name and an index.

        Returns:
            save_dir (Path): Path to the save directory.
        """
        results_id = 1
        experiment_name = self.experiment_name
        save_dir = Path(
            f"results/{experiment_name}_{socket.gethostname()}_{results_id}"
        )

        while Path(save_dir).exists():
            results_id += 1
            save_dir = Path(
                f"results/{experiment_name}_{socket.gethostname()}_{results_id}"
            )
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _setup_logger(self):
        # Implement your logger setup here
        logger = logging.getLogger("experiment_logger")
        logger.setLevel(logging.INFO)
        # Add handlers, formatters, etc.
        return logger

    def _save_config(self, config):
        config_filename = f"{self.experiment_name}_config.yaml"
        with open(self.save_dir / config_filename, "w") as f:
            yaml.dump(config, f)
        self.logger.info(
            f"SAVED Config file [{config_filename}] to [{self.save_dir}].\n"
        )

    @abstractmethod
    def load_data(self):
        """Load the data required for the experiment."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Preprocess the data before training."""
        pass

    @abstractmethod
    def train_model(self):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate_model(self):
        """Evaluate the model."""
        pass

    @abstractmethod
    def save_results(self):
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
