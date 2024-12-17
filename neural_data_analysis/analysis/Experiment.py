#!/usr/bin/env python3

import socket
from datetime import datetime
from pathlib import Path
import logging

import yaml
from ..utils import add_default_repr

from abc import ABC, abstractmethod
from ..utils import setup_logger, setup_default_logger

from ..models import LogisticModelWrapper, MLPModelWrapper, LSTMModelWrapper
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from scipy.special import expit
import numpy as np

example_config = {
    "project_name": "example_project",
    "cross_validation": {
        "n_folds": 5,
        "kfolds_stratify": True,
        "kfolds_shuffle": True,
    },
    "model_configs": {
        "LinearModel": {
            "problem_type": "binary_classification",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
        },
        "MLPModel": {
            "problem_type": "multiclass_classification",
            "class_weight": "balanced",
            "solver": "adam",
            "max_iter": 1000,
        },
        "XGBModel": {
            "problem_type": "binary_classification",
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 1000,
        },
    },
}


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
        """ """
        # Create experiment name and save directory
        self.experiment_name = self._create_experiment_name()
        self.save_dir = self._create_save_dir()
        CLASS_NAME = self.__class__.__name__
        self.logger = setup_logger(
            logger_name=CLASS_NAME,
            log_filepath=Path(f"{self.save_dir}/{self.experiment_name}.log"),
        )

        self.logger.info(f"========== Initializing {CLASS_NAME} class ==========")
        self.logger.info(
            f"LOADED: Config file [{self.config['general']['config_file']}] in ExperimentRunner class."
        )
        self.logger.info(f"CREATED: Save directory [{str(self.save_dir)}].")
        self._save_config(self.config)

    def _create_experiment_name(self) -> str:
        """
        Create an experiment name based on the date and project name.

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
            f"SAVED: Experiment config file [{config_filename}] to save directory [{self.save_dir}].\n"
        )

    @abstractmethod
    def _load_data(self):
        """Load the data required for the experiment."""
        pass

    def _generate_splits(
        self,
        indices_to_split: np.ndarray,
        cross_validation_config: dict,
        y: np.ndarray = None,
    ):
        """Generate the splits for KFold or StratifiedKFold.

        Args:
            indices_to_split (np.ndarray): indices for the data to split into folds
            cross_validation_config (dict): configuration for the cross-validation
            y (np.ndarray): the class labels to use for class balanced StratifiedKFold splits

        Returns:
            the generator class that returns the splits
        """
        # Decide which class to use based on the 'kfolds_stratify' config
        stratify_bool = cross_validation_config.get("kfolds_stratify", False)
        n_folds = cross_validation_config.get("n_folds")
        shuffle_bool = cross_validation_config.get("kfolds_shuffle", False)

        KFoldClass = MultilabelStratifiedKFold if stratify_bool else KFold

        # Create the fold generator with appropriate configuration
        fold_generator = KFoldClass(
            n_splits=n_folds,
            shuffle=shuffle_bool,
            random_state=(
                self.config["ExperimentRunner"]["seed"] if shuffle_bool else None
            ),
        )

        # Generate and return the splits
        if stratify_bool:
            return fold_generator.split(indices_to_split, y)
        else:
            return fold_generator.split(indices_to_split)

    def _get_train_val_data(self, X, Y, train_index, val_index):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = Y[train_index], Y[val_index]
        return X_train, X_val, y_train, y_val

    def _initialize_model(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        model_type: str,
        model_configs: dict,
    ):
        """
        Initialize the model class based on the model type.

        Args:
            X (np.ndarray): input data
            y (np.ndarray): output data
            model_type (str): model type to initialize

        Returns:
            Class of the model_type.
        """
        input_dims = X.shape[1]
        output_dims = Y.shape[1] if len(Y.shape) > 1 else 1

        if model_type == "logistic":
            model_config = model_configs.get("LogisticModel")
            model = LogisticModelWrapper(model_config)
        elif model_type == "mlp":
            model_config = model_configs.get("MLPModel")
            model_config["input_dims"] = input_dims
            model_config["output_dims"] = output_dims

            if model_config.get("use_pos_weights", False):
                # calculate weights for imbalanced classes, setting to 1 if all samples are positive or negative
                # ratio of negative class labels to positive labels. If negative labels are 2:1,
                # then assigns a weight of 2 to the sparser positive labels
                with np.errstate(divide="ignore", invalid="ignore"):
                    pos_weights = np.sum(Y == 0, axis=0) / np.sum(Y, axis=0)
                pos_weights = np.nan_to_num(pos_weights, nan=1, posinf=1, neginf=1)
                pos_weights[pos_weights == 0] = 1
                # assign these weights to the config
                model_config["pos_weights"] = pos_weights

            model = self._initialize_mlp_model(model_config, X, Y)
        elif model_type == "xgb":
            model_config = model_configs.get("XGBModel")
            model = self._initialize_xgb_model(model_config)
        elif model_type == "lstm":
            model_config = model_configs.get("LSTMModel")
            model_config["input_dims"] = input_dims
            model_config["output_dims"] = output_dims

            if model_config.get("use_pos_weights", False):
                # calculate weights for imbalanced classes, setting to 1 if all samples are positive or negative
                # ratio of negative class labels to positive labels. If negative labels are 2:1,
                # then assigns a weight of 2 to the sparser positive labels
                with np.errstate(divide="ignore", invalid="ignore"):
                    pos_weights = np.sum(Y == 0, axis=0) / np.sum(Y, axis=0)
                pos_weights = np.nan_to_num(pos_weights, nan=1, posinf=1, neginf=1)
                pos_weights[pos_weights == 0] = 1
                # assign these weights to the config
                model_config["pos_weights"] = pos_weights

            model = LSTMModelWrapper(model_config)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")
        return model, model_config

    def _initialize_mlp_model(self, mlp_config: dict, X, Y):
        problem_type = mlp_config.get("problem_type")

        if problem_type == "binary_classification":
            assert len(np.unique(Y)) == 2
        elif problem_type == "multiclass_classification":
            n_classes = len(np.unique(Y))
            assert n_classes > 2
            output_dim = n_classes
            mlp_config["output_dim"] = output_dim
        elif problem_type == "regression":
            pass
        else:
            raise ValueError(f"Problem type {problem_type} not recognized.")

        model = MLPModelWrapper(mlp_config)
        model_class_name = model.model.__class__.__name__
        self.logger.info(
            f"Model type: [{model_class_name}], Problem type: [{mlp_config['problem_type']}]."
        )
        return model

    def _initialize_xgb_model(self, xgb_config):
        if xgb_config["problem_type"] == "regression":
            # Implement regression model initialization
            pass
        elif xgb_config["problem_type"] in ["classification", "binary_classification"]:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MultiOutputClassifier(
                            xgb.XGBClassifier(
                                objective="binary:logistic",
                                max_depth=xgb_config["max_depth"],
                                n_estimators=xgb_config["n_estimators"],
                                learning_rate=xgb_config["learning_rate"],
                                n_jobs=-1,
                                max_leaves=xgb_config["max_leaves"],
                                random_state=self.config["seed"],
                                eval_metric="logloss",
                            )
                        ),
                    ),
                ]
            )
            return model
        else:
            raise ValueError(
                "XGBModel problem type not recognized. Must be classification or regression."
            )

    def _model_fit(
        self, model, X_train, y_train, X_val, y_val, run_info: dict, model_type: str
    ):
        model_class_name = model.model.__class__.__name__
        self.logger.info(f"Fitting model of class [{model_class_name}]...")
        if model_type == "linear":
            model.fit(X_train, y_train)
        elif model_type == "mlp":
            model.fit(X_train, y_train, X_val, y_val, run_info)
        elif model_type == "xgb":
            model.fit(X_train, y_train)
        elif model_type == "lstm":
            # input needs to be (batch_size, timesteps/seq_length, input_dim/feature_size)
            X_train_lstm = X_train[:, np.newaxis, :]
            X_val_lstm = X_val[:, np.newaxis, :]
            model.fit(X_train_lstm, y_train, X_val_lstm, y_val, run_info)
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

    def _model_predict(self, model, X_val, model_config: dict, model_type: str):
        model_predictions = model.predict(X_val)
        if model_type == "mlp":
            predictions = self._compute_mlp_predictions(
                predictions=model_predictions, mlp_config=model_config
            )
        else:
            raise ValueError(f"Model type {model_type} not recognized.")
        return predictions

    def _compute_mlp_predictions(self, predictions, mlp_config: dict):
        if mlp_config["problem_type"] == "multi_class_classification":
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)
        elif mlp_config["problem_type"] == "binary_classification":
            # numpy does not have sigmoid function like torch, so we use the equivalent scipy.special.expit function
            predictions = (expit(predictions) > 0.5).astype(int)
        else:
            raise ValueError(
                f"Problem type {mlp_config['problem_type']} not recognized."
            )
        return predictions

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
