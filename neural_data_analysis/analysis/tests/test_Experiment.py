import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
import socket
import logging

from neural_data_analysis.analysis import Experiment


class ConcreteExperiment(Experiment):
    """
    Need to implement the abstract methods in the Experiment class.
    """

    def load_data(self):
        pass

    def preprocess_data(self):
        pass

    def train_model(self):
        pass

    def evaluate_model(self):
        pass

    def save_results(self):
        pass


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.sample_config = {
            "general": {"config_file": "test_config.yaml"},
            "ExperimentRunner": {
                "project_name": "test_project",
            },
        }

    @patch("neural_data_analysis.analysis.Experiment.datetime")
    @patch("neural_data_analysis.analysis.Experiment.socket.gethostname")
    @patch("neural_data_analysis.analysis.Experiment.setup_logger")
    @patch("neural_data_analysis.analysis.Experiment.setup_default_logger")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    def test_initialization_creates_save_dir_and_logs(
        self,
        mock_open_fn,
        mock_mkdir,
        mock_setup_default_logger,
        mock_setup_logger,
        mock_gethostname,
        mock_datetime,
    ):
        # Mock datetime to return a fixed date
        mock_datetime.now.return_value.strftime.return_value = "2021-01-01"

        # Mock gethostname to return a fixed hostname
        mock_gethostname.return_value = "test_host"

        # Mock logger
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        # Instantiate the concrete subclass
        experiment = ConcreteExperiment(config=self.sample_config, load_only=False)

        # Assertions
        self.assertIsNotNone(experiment.experiment_name)
        self.assertIsNotNone(experiment.save_dir)
        mock_mkdir.assert_called_once()
        mock_open_fn.assert_called()  # Check that open was called to save the config
        mock_logger.info.assert_any_call(
            f"LOADED: Config file [{self.sample_config['general']['config_file']}] in ExperimentRunner class."
        )
        mock_logger.info.assert_any_call(
            f"COMPLETED: Created save directory [{str(experiment.save_dir)}]."
        )

    @patch("neural_data_analysis.analysis.Experiment.setup_logger")
    @patch("neural_data_analysis.analysis.Experiment.setup_default_logger")
    def test_load_only_mode(self, mock_setup_default_logger, mock_setup_logger):
        # Mock logger
        mock_logger = MagicMock()
        mock_setup_default_logger.return_value = mock_logger

        # Instantiate the class with load_only=True
        experiment = ConcreteExperiment(config=self.sample_config, load_only=True)

        # Assertions
        self.assertFalse(hasattr(experiment, "experiment_name"))
        self.assertFalse(hasattr(experiment, "save_dir"))
        mock_setup_default_logger.assert_called_once()
        mock_setup_logger.assert_not_called()

    @patch("neural_data_analysis.analysis.Experiment.datetime")
    def test_experiment_name_creation(self, mock_datetime):
        # Mock datetime to return a fixed date
        mock_datetime.now.return_value.strftime.return_value = "2021-01-01"

        experiment = Experiment(config=self.sample_config, load_only=True)

        # Call the method
        experiment_name = experiment._create_experiment_name()

        expected_name = "2021-01-01_test_project"
        self.assertEqual(experiment_name, expected_name)

    @patch("neural_data_analysis.analysis.Experiment.socket.gethostname")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_create_save_dir(self, mock_mkdir, mock_exists, mock_gethostname):
        # Mock hostname
        mock_gethostname.return_value = "test_host"

        # Simulate that the directory does not exist
        mock_exists.return_value = False

        experiment = ConcreteExperiment(config=self.sample_config, load_only=True)
        experiment.experiment_name = "2021-01-01_test_project"

        # Call the method
        save_dir = experiment._create_save_dir()

        expected_dir = Path("results/2021-01-01_test_project_test_host_1")
        self.assertEqual(save_dir, expected_dir)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("neural_data_analysis.analysis.Experiment.socket.gethostname")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    def test_create_save_dir_directory_exists(
        self, mock_mkdir, mock_exists, mock_gethostname
    ):
        # Mock hostname
        mock_gethostname.return_value = "test_host"

        # Simulate that the directory exists on first check, then doesn't
        mock_exists.side_effect = [True, False]

        experiment = ConcreteExperiment(config=self.sample_config, load_only=True)
        experiment.experiment_name = "2021-01-01_test_project"

        # Call the method
        save_dir = experiment._create_save_dir()

        expected_dir = Path("results/2021-01-01_test_project_test_host_2")
        self.assertEqual(save_dir, expected_dir)
        self.assertEqual(mock_exists.call_count, 2)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("builtins.open", new_callable=mock_open)
    @patch("neural_data_analysis.analysis.Experiment.yaml.dump")
    def test_save_config(self, mock_yaml_dump, mock_open_fn):
        # Mock logger
        mock_logger = MagicMock()

        experiment = ConcreteExperiment(config=self.sample_config, load_only=True)
        experiment.logger = mock_logger
        experiment.experiment_name = "2021-01-01_test_project"
        experiment.save_dir = Path("/fake/dir")

        # Call the method
        experiment._save_config(experiment.config)

        expected_filename = (
            experiment.save_dir / f"{experiment.experiment_name}_config.yaml"
        )
        mock_open_fn.assert_called_once_with(expected_filename, "w")
        mock_yaml_dump.assert_called_once_with(experiment.config, mock_open_fn())
        mock_logger.info.assert_called_once_with(
            f"SAVED Config file [{experiment.experiment_name}_config.yaml] to [{experiment.save_dir}].\n"
        )


if __name__ == "__main__":
    unittest.main()
