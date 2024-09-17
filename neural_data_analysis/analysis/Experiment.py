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


# noinspection PyShadowingNames


@add_default_repr
class Experiment:
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

        self.config: dict = config["ExperimentRunner"]

        # whether to create a save directory and log file
        if load_only:
            self.logger = setup_default_logger()
        else:
            # Create a save directory for the experiment results.
            self.experiment_name: str = self.create_experiment_name()
            self.save_dir: Path = self.create_save_dir()

            # Create a logger object to log the experiment.
            setup_logger("logger", Path(f"{self.save_dir}/{self.experiment_name}.log"))
            self.logger = logging.getLogger("logger")

            # Save the config file to the results folder. Requires a logger to have been created.
            self._save_config(config)

            # Log the configuration file and the save directory.
            self.logger.info(
                f"LOADED: Config file [{config['general']['config_file']}] in ExperimentRunner class."
            )
            self.logger.info(
                f"COMPLETED: Created save directory [{str(self.save_dir)}]."
            )

        # Get the neural and video data that will be used for the experiment based on the target variables in the config file.
        self.neural_data: dict = self.get_neural_variable(
            self.config["neural_target_variable"]
        )
        self.video_data: dict = self.get_video_variable(
            self.config["video_target_variable"]
        )

    def create_experiment_name(self):
        project_name = self.config.get("project_name", "project")
        experiment_name = self.config.get("experiment_name", "experiment")
        date = datetime.now().strftime("%Y-%m-%d")
        full_experiment_name = f"{date}_{project_name}_{self.config['type']}_{self.config['model']}_{experiment_name}"
        return full_experiment_name

    def create_save_dir(self) -> Path:
        """Create a save directory for the results.

        The directory name is made up of the experiment name and an index.

        Returns:
            save_dir (Path): Path to the save directory.

        """
        results_id = 1
        experiment_name = self.experiment_name
        save_dir = Path(
            f"results/{experiment_name}_{socket.gethostname()}_{results_id}"
        )
        if Path(save_dir).exists():
            while Path(save_dir).exists():
                results_id += 1
                save_dir = Path(
                    f"results/{experiment_name}_{socket.gethostname()}_{results_id}"
                )
        save_dir.mkdir()
        return save_dir

    def get_video_variable(self, var_name: str = "raw_embedding") -> dict:
        """Return the video feature to be used.

        Args: var_name (str): The feature to be used. Options are "raw_embedding", "pca_embedding", "kmeans_cluster",
        "shot_scene".

        Returns:
            data (dict): The video feature to be used as a dictionary.

        """
        if var_name == "raw_embeddings":
            data = self.video_data_loader.embeddings
        elif var_name == "pca_embedding":
            self.video_data_loader.embeddings_pca = (
                self.video_data_loader.create_frame_embeddings_pca()
            )
            data = self.video_data_loader.embeddings_pca
        elif var_name == "kmeans_cluster":
            self.video_data_loader.cluster_labels = (
                self.video_data_loader.kmeans_clustering(
                    seed=self.config["seed"],
                    dim_reduction_method=self.video_data_loader.config[
                        "dim_reduction_method"
                    ],
                    cumulative_variance=self.video_data_loader.config[
                        "cumulative_variance"
                    ],
                    n_clusters=self.video_data_loader.config["n_kmeans_clusters"],
                    plot_dir=Path(f"{self.save_dir}/plots"),
                )
            )
            data = self.video_data_loader.cluster_labels
        elif var_name == "shot_scene":
            data = self.video_data_loader.shot_scene_labels
        else:
            raise ValueError(
                "Video variable type not recognized. Must be one of: raw_embedding, pca_embedding, kmeans_cluster, "
                "shot_scene"
            )
        return data

    def get_neural_variable(self, var_name: str = "population_fr") -> dict:
        if var_name == "population_fr":
            data = self.neural_data_loader.population_fr["population_fr"]
        elif var_name == "combined_timepoints":
            # TODO: debug this!
            self.neural_data_loader.population_fr_combined = (
                self.neural_data_loader.combine_fr_timepoints()
            )
            data = self.neural_data_loader.population_fr_combined
        else:
            raise ValueError(
                "Neural variable type not recognized. Must be one of: [population_fr, combined_timepoints]"
            )
        return data

    def block_frames(self) -> dict:
        """
        Splits the movie into blocks of frames. Each block should only contain frames from a single shot.

        Arguments:


        Returns:
            blocks_dict (dict): dictionary of blocks.
                Keys are block indices, values are the indices of the frames in the block.
        """
        # block_by (str): whether to block by "shots" or "frames"
        # by "shots" means that each block will contain all frames from a single shot
        # by "frames" means that each block will contain a fixed number of frames from within a shot
        block_by = self.config["block_frames"]["block_by"]

        if block_by == "frames":
            blocks_dict = self._block_by_frames()
        elif block_by == "shots":
            blocks_dict = self._block_by_shots()

        elif block_by == "frames_ignore_shots":
            # ignores what shot a frame belongs to and just groups by block_size
            idx_arr = np.arange(len(self.video_data_loader.frames))
            block_size = self.config["block_frames"]["block_size"]
            blocks = np.array(
                [
                    idx_arr[i : i + block_size]
                    for i in range(0, len(idx_arr), block_size)
                ]
            )
            blocks_dict = {i: block for i, block in enumerate(blocks)}
            self.logger.info(
                f"Frames blocked by [{block_by}] with block size [{block_size}]. Frames from different shots may "
                f"appear in the same block."
            )
        else:
            raise ValueError(f"block_by type [{block_by}] not recognized.")

        return blocks_dict

    # noinspection PyPep8Naming
    def compute_shap_values(
        self,
        model,
        shap_explainer: str = "base",
        X_train: np.ndarray = np.empty(0),
        X_val: np.ndarray = np.empty(0),
    ) -> np.ndarray:
        """
        Computes the SHAP values for the validation set.

        This function uses the provided model to calculate SHAP values, which
        are measures of feature importance, for each input feature in the validation dataset.

        Parameters:
            model: The model to use for computing SHAP values. Must have predict() function.
            shap_explainer (str): Type of SHAP explainer to use. Options are "kernel" or "deep".
            X_train (np.ndarray(): Shape (n_samples, n_inputs)
                Number of training samples by number of inputs into the model.
            X_val (np.ndarray): Shape (n_samples, n_inputs)
                Number of validation samples by number of inputs into the model.

        Returns:
            shap_values (np.ndarray): (n_inputs, n_outputs)
        """
        self.logger.info(f"--------------- Computing SHAP values -----------------")
        self.logger.info(
            f"Model type: [{self.config['model']}], SHAP explainer: [{shap_explainer}]."
        )
        # the number of validation samples should be at least the number of features if using KernelExplainer
        background_nsamples = 100  # or X_train.shape[0]
        validation_nsamples = X_val.shape[0]  # or X_val.shape[0]
        # This is the number of subset samples to use to estimate the SHAP values for KernelExplainer
        shap_nsamples = "auto"  # or "auto"
        self.logger.info(f"Number of background samples: [{background_nsamples}].")
        self.logger.info(f"Number of validation samples: [{validation_nsamples}].")
        self.logger.info(f"Number of SHAP samples: [{shap_nsamples}].")

        model_type = self.config["model"]
        # When masking out features to say that they are missing, we need to provide a background dataset to replace
        # those features. Background samples used for baseline values of features, replacements for marginalizing out
        # features, and for the expected value calculation of the model. documentation says that 100 samples is
        # enough for a good estimate
        background = X_train[
            np.random.choice(X_train.shape[0], background_nsamples, replace=False)
        ]
        self.logger.info(f"The shape of the background data is {background.shape}.")
        # Select a smaller subset of the validation data for which to compute SHAP values
        # sample_size = X_val.shape[0]  # Adjust this as needed
        X_val_sample = X_val[
            np.random.choice(X_val.shape[0], validation_nsamples, replace=False)
        ]
        self.logger.info(f"The shape of the validation data is {X_val_sample.shape}.")

        # Compute SHAP values
        shap_values = np.empty(0)
        if model_type == "linear":
            if shap_explainer == "base":
                explainer = shap.Explainer(
                    model.model.predict,
                    X_train,
                    feature_names=self.video_data_loader.blip2_words_df[
                        "word"
                    ].tolist(),
                )
                shap_values = explainer(X_val)
                # shap.plots.beeswarm(shap_values)
            elif shap_explainer == "linear":
                if isinstance(model.model, Pipeline):
                    scaler = model.model.named_steps["scaler"]
                    classifier = model.model.named_steps["classifier"]
                    background = scaler.transform(background)
                    explainer = shap.LinearExplainer(classifier, background)
                    X_val_sample = scaler.transform(X_val_sample)
                    shap_values = explainer.shap_values(X_val_sample)
            elif shap_explainer == "kernel":
                explainer = shap.KernelExplainer(model.model.predict, background)
                shap_values = explainer.shap_values(
                    X_val_sample, nsamples=shap_nsamples
                )
            elif shap_explainer == "permutation":
                max_evals = 2 * X_train.shape[1] + 1  # Calculate the required max_evals
                explainer = shap.PermutationExplainer(
                    model.model.predict, X_train, max_evals=max_evals
                )
                shap_values = explainer.shap_values(X_val_sample)
            else:
                raise ValueError(
                    f"SHAP explainer type [{shap_explainer}] not recognized for model type [{model_type}]."
                )
        elif model_type == "mlp":
            # TODO: the kernel explainer is still in progress and not working
            if shap_explainer == "kernel":
                assert X_val_sample.shape[0] > X_val_sample.shape[1], (
                    f"For [{shap_explainer}] SHAP explainer, the number of samples [{X_val_sample.shape[0]}] must not "
                    f"be less than the number of features [{X_val_sample.shape[1]}]"
                )
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(
                    X_val_sample, nsamples=shap_nsamples
                )
            elif shap_explainer == "deep":
                model_for_shap = model.model
                model_for_shap.eval()
                # model_for_shap.to(model.device)
                # .float() converts to float32 which is the data type that the model weights are in
                # background = torch.tensor(background).to(model.device).float()
                # X_val_sample = torch.tensor(X_val_sample).to(model.device).float()
                background = torch.tensor(background).float()
                X_val_sample = torch.tensor(X_val_sample).float()
                explainer = shap.DeepExplainer(model_for_shap, background)
                # input should be (n_samples, n_features)
                shap_values = explainer.shap_values(X_val_sample)
                # output is (n_samples, n_inputs, n_outputs)
                shap_values = np.array(shap_values).transpose([2, 0, 1])
            else:
                raise ValueError(
                    f"SHAP explainer type [{shap_explainer}] not recognized for model type [{model_type}]."
                )
        elif model_type == "xgb":
            if shap_explainer == "tree":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val_sample)
            else:
                explainer = shap.Explainer(model)
                shap_values = explainer(X_val_sample)
        else:
            raise ValueError("Model type not recognized.")

        self.logger.info(
            f"SHAP values computed. Shape is {shap_values.shape}.\n"
            f"This should be (n_outputs, n_samples, n_inputs) for a multioutput model.\n"
            f"Or (n_samples, n_inputs) for a single output model."
        )
        if shap_values.ndim == 2:
            self.logger.info(
                f"Averaging SHAP values across n_samples {shap_values.shape[0]}."
            )
            shap_values = np.mean(shap_values, axis=0).reshape(-1, 1)
        else:
            # output of explainer is shape (n_outputs, n_samples, n_inputs)
            # reformat to shape (n_samples, n_inputs, n_outputs)
            self.logger.info(
                "Reshaping the SHAP values to (n_samples, n_inputs, n_outputs)..."
            )
            shap_values = np.array(shap_values).transpose([1, 2, 0])
            self.logger.info(f"SHAP values dimensions are now {shap_values.shape}.")
            self.logger.info(
                f"Averaging SHAP values across n_samples {shap_values.shape[0]}."
            )
            # TODO: also implement summing the SHAP values
            # average across n_samples, shape is (n_inputs, n_outputs)
            shap_values = np.mean(shap_values, axis=0)
        self.logger.info(
            f"Final SHAP values dimensions are {shap_values.shape}. This is (n_inputs, n_outputs)."
        )
        return shap_values

    def record_shap_details(
        self, shap_values: np.ndarray, fold_idx: int, neural_key, video_key
    ) -> list[list]:
        """Organizes the shape values into a list of lists with information about the cell, brain area, and fold.

        Args:
            shap_values:
            fold_idx:
            neural_key:
            video_key:

        Returns:
            shap_details (list[list]): A list of lists with the following information:
                - neural_key (tuple): The brain area, bin center, and bin size.
                - video_key (str): The video feature.
                - fold_idx (int): The fold index.
                - cell_info (tuple): The cell id, cluster id, and importance.
                - brain_areas (str): The brain area.
                - output_idx (int): The output index.
                - output (float): The output value.

        """
        brain_areas = self.neural_data_loader.population_fr["brain_areas"][neural_key]
        cell_info = self.neural_data_loader.population_fr["cell_info"][neural_key]
        assert len(brain_areas) == len(cell_info) == shap_values.shape[0]
        shap_details = []
        # TODO: this doesn't work for 2 classes... perhaps because that's binary classification instead of multiclass...
        for input_idx, model_input in enumerate(shap_values):
            for output_idx, model_output in enumerate(model_input):
                shap_details.append(
                    [
                        neural_key,
                        video_key,
                        fold_idx,
                        cell_info[input_idx],
                        brain_areas[input_idx],
                        output_idx,
                        model_output,
                    ]
                )
        return shap_details

    # noinspection PyPep8Naming
    def run_experiment(self) -> pd.DataFrame:
        self.logger.info("=============== Running Experiment ===============")

        # lists to save results for each experiment
        results = []
        importances = []
        for neural_key, video_key in itertools.product(
            self.neural_data.keys(),
            self.video_data.keys(),
        ):
            self.logger.info(
                f"NEURAL DATA: BRAIN AREA [{neural_key[0]}], BIN CENTER [{neural_key[1]}], BIN SIZE [{neural_key[2]}], "
                f"[{self.config['neural_target_variable']}]"
            )
            self.logger.info(
                f"VIDEO DATA: [{self.config['video_target_variable']}] of [{video_key}]"
            )

            results_partial = []
            importances_partial = []

            if self.config["model_type"] == "decoding":
                X = self.neural_data[neural_key]
                Y = self.video_data[video_key]
                if self.config["problem_type"] == "binary_classification":
                    Y = Y.astype(int)
                self.logger.info(
                    f"Running [{self.config['problem_type']} {self.config['model_type']}] experiment to predict VIDEO "
                    f"FEATURES from NEURAL DATA."
                )
                assert (
                    X.shape[0] == Y.shape[0]
                ), "X and Y must have the same number of samples."
            else:
                raise ValueError("Experiment type not recognized.")

            # ------------------------------ K-Fold Cross Validation ------------------------------

            # create the indices to split the data into folds
            # whether to group the frames into blocks
            blocks = None
            if self.config["block_frames"]["do_blocking"]:
                blocks = self.block_frames()
                indices_to_split = list(blocks.keys())
            else:
                indices_to_split = np.arange(X.shape[0])
                self.logger.info(
                    "No blocking of frames. Splitting into KFolds with individual frames."
                )
            # TODO: Cannot do blocking AND do stratified KFold. Need to fix this.
            # TODO: this also does not handle multilabel classification
            split_generator = self._generate_splits(indices_to_split, Y)

            # start K fold cross validation
            fold_idx = 0
            for train_index, val_index in tqdm(
                split_generator,
                desc="Cross-Validation Folds",
                bar_format="{l_bar}{bar}{r_bar}\n",
            ):
                self.logger.info(
                    f"================ Fold: {fold_idx + 1} out of {self.config['n_folds']} ================="
                )

                run_info = {
                    "save_dir": Path(self.save_dir),
                    "brain_area": neural_key[0],
                    "bin_center": neural_key[1],
                    "bin_size": neural_key[2],
                    "video_var": video_key,
                    "fold": fold_idx,
                    "run_name": f"{self.experiment_name}_{neural_key[0]}_{neural_key[1]}_{neural_key[2]}"
                    f"_{video_key}_{fold_idx}",
                }

                if self.config["block_frames"]["do_blocking"]:
                    # unpack the blocks back into individual frames
                    # the blocks don't have the same number of frames
                    # this does mean that the folds will have an uneven number of samples
                    train_index = np.concatenate(
                        [blocks[key] for key in train_index]
                    ).astype(int)
                    val_index = np.concatenate(
                        [blocks[key] for key in val_index]
                    ).astype(int)

                X_train, X_val = (
                    X[train_index],
                    X[val_index],
                )
                y_train, y_val = (
                    Y[train_index],
                    Y[val_index],
                )
                self.logger.info(
                    f"The shape of y_train is {y_train.shape}. The number of unique values in y_train is "
                    f"{len(np.unique(y_train))}."
                )
                self.logger.info(
                    f"The shape of y_val is {y_val.shape}. The number of unique values in y_val is "
                    f"{len(np.unique(y_val))}."
                )

                model_type = self.config["model"]
                # ------------------------------ Linear Model ------------------------------

                if model_type == "linear":
                    linear_config = self.config["LinearModel"]
                    model = LinearModelWrapper(linear_config)

                # ------------------------------ Multilayer Perceptron ------------------------------

                elif model_type == "mlp":
                    mlp_config = self.config["MLPModel"]
                    # change the output dimensions depending on the problem type
                    if mlp_config["problem_type"] == "regression":
                        model = MLPModelWrapper(
                            mlp_config, X_train.shape[-1], y_val.shape[-1]
                        )
                    elif mlp_config["problem_type"] == "multi_class_classification":
                        n_classes = len(np.unique(Y))
                        model = MLPModelWrapper(
                            mlp_config, X_train.shape[-1], n_classes
                        )
                    elif mlp_config["problem_type"] == "binary_classification":
                        n_classes = len(np.unique(Y))
                        assert n_classes == 2
                        n_labels = y_val.shape[-1]
                        model = MLPModelWrapper(mlp_config, X_train.shape[-1], n_labels)
                    else:
                        raise ValueError(
                            f"MLPModel problem type [{mlp_config['problem_type']}] not recognized. Must be "
                            f"classification or regression."
                        )
                    self.logger.info(
                        f"Model type: [{type(model)}], Problem type: [{mlp_config['problem_type']}]."
                    )

                # ------------------------------ Gradient Boosted Trees ------------------------------
                elif model_type == "xgb":
                    xgb_config = self.config["XGBModel"]
                    if xgb_config["problem_type"] == "regression":
                        raise NotImplementedError
                    elif (
                        xgb_config["problem_type"] == "classification"
                        or "multi_class_classification"
                        or "binary_classification"
                    ):
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

                    else:
                        raise ValueError(
                            "XGBModel problem type not recognized. Must be classification or regression."
                        )

                else:
                    raise ValueError(f"Model type {model_type} not recognized.")

                # ------------------------------ Fit Model ------------------------------
                self.logger.info(f"Fitting model of class [{type(model)}]...")
                if model_type == "linear":
                    model.fit(X_train, y_train)
                elif model_type == "mlp":
                    model.fit(X_train, y_train, X_val, y_val, run_info)
                elif model_type == "xgb":
                    model.fit(X_train, y_train)
                else:
                    raise ValueError(f"Model type {model_type} not recognized.")

                # ------------------------------ Predictions ------------------------------
                ground_truth = y_val
                predictions = model.predict(X_val)
                if model_type == "linear":
                    linear_config = self.config["LinearModel"]
                    if linear_config["problem_type"] == "binary_classification":
                        self.logger.info(
                            f"Balanced Accuracy Score: {balanced_accuracy_score(ground_truth, predictions)}"
                        )
                        self.logger.info(
                            f"Accuracy Score: {accuracy_score(ground_truth, predictions)}"
                        )
                        self.logger.info(
                            f"Precision Score: {precision_score(ground_truth, predictions)}"
                        )
                        self.logger.info(
                            f"Recall Score: {recall_score(ground_truth, predictions)}"
                        )
                        self.logger.info(
                            f"F1 Score: {f1_score(ground_truth, predictions)}"
                        )

                    elif linear_config["problem_type"] == "multi_class_classification":
                        # Note: In multilabel classification, accuracy_score computes subset accuracy: the set of labels
                        # predicted for a sample must exactly match the corresponding set of labels in y_true.
                        self.logger.info(
                            f"Accuracy Score: {accuracy_score(ground_truth, predictions)}"
                        )
                        self.logger.info(
                            f"Hamming Loss: {hamming_loss(ground_truth, predictions)}"
                        )
                if model_type == "mlp":
                    mlp_config = self.config["MLPModel"]
                    if mlp_config["problem_type"] == "multi_class_classification":
                        if not len(predictions.shape) == 1:
                            predictions = np.argmax(predictions, axis=1)
                    elif mlp_config["problem_type"] == "binary_classification":
                        predictions = (expit(predictions) > 0.5).astype(int)

                results.append(
                    [
                        run_info["brain_area"],
                        run_info["bin_center"],
                        run_info["bin_size"],
                        run_info["video_var"],
                        run_info["fold"],
                        self.video_data_loader.original_frame_index[val_index],
                        ground_truth,
                        predictions,
                    ]
                )
                results_partial.append(
                    [
                        run_info["brain_area"],
                        run_info["bin_center"],
                        run_info["bin_size"],
                        run_info["video_var"],
                        run_info["fold"],
                        self.video_data_loader.original_frame_index[val_index],
                        ground_truth,
                        predictions,
                    ]
                )
                # ------------------------------ Compute Importances ------------------------------
                if self.config["compute_importances"]:
                    self.logger.info("Computing importances...")
                    shap_values = self.compute_shap_values(
                        model,
                        shap_explainer=self.config["shap_explainer"],
                        X_train=X_train,
                        X_val=X_val,
                    )
                    shap_details = self.record_shap_details(
                        shap_values,
                        fold_idx,
                        neural_key,
                        video_key,
                    )
                    importances.extend(shap_details)
                    importances_partial.extend(shap_details)
                fold_idx += 1

            # save partial results for each iteration over the neural and video parameters
            # noinspection DuplicatedCode
            results_partial_df = pd.DataFrame(
                results_partial,
                columns=[
                    "brain_area",
                    "bin_center",
                    "bin_size",
                    "embedding",
                    "fold",
                    "original_frame_index",
                    "ground_truth",
                    "predictions",
                ],
            ).astype(
                {
                    "brain_area": "string",
                    "bin_center": "float",
                    "bin_size": "float",
                    "embedding": "string",
                    "fold": "int",
                }
            )
            results_partial_df["ground_truth"] = results_partial_df[
                "ground_truth"
            ].apply(np.array)
            results_partial_df["predictions"] = results_partial_df["predictions"].apply(
                np.array
            )
            self._save_results(
                results_partial_df,
                name=f"results_partial_{'_'.join(map(str, neural_key))}_{video_key}",
                save_type="pickle",
            )
            if self.config["compute_importances"]:
                importances_partial_df = pd.DataFrame(
                    importances_partial,
                    columns=[
                        "neural",
                        "embedding",
                        "fold",
                        "cell_id",
                        "brain_area",
                        "output",
                        "importance",
                    ],
                )
                self.logger.info("Importance results converted to DataFrame.")
                self._save_results(
                    importances_partial_df,
                    name=f"shap_values_{'_'.join(map(str, neural_key))}_{video_key}",
                    save_type="csv",
                )
            else:
                self.logger.info("No importance results to save.")

        # noinspection DuplicatedCode
        results_df = pd.DataFrame(
            results,
            columns=[
                "brain_area",
                "bin_center",
                "bin_size",
                "embedding",
                "fold",
                "original_frame_index",
                "ground_truth",
                "predictions",
            ],
        ).astype(
            {
                "brain_area": "string",
                "bin_center": "float",
                "bin_size": "float",
                "embedding": "string",
                "fold": "int",
            }
        )
        results_df["ground_truth"] = results_df["ground_truth"].apply(np.array)
        results_df["predictions"] = results_df["predictions"].apply(np.array)
        print(results_df.dtypes)
        self.logger.info("Results converted to DataFrame.")
        # results_df.to_hdf("results.h5", key="results1", mode="w")

        # save results
        self._save_results(results_df, name="results", save_type="pickle")

        if self.config["compute_importances"]:
            importances_df = pd.DataFrame(
                importances,
                columns=[
                    "neural",
                    "embedding",
                    "fold",
                    "cell_id",
                    "brain_area",
                    "output",
                    "importance",
                ],
            )
            self.logger.info("Importance results converted to DataFrame.")
            self._save_results(importances_df, name="shap_values", save_type="csv")
        else:
            self.logger.info("No importance results to save.")
        self.logger.info(
            "=========================== COMPLETED: Experiment Done. ==============================\n"
        )
        return results_df

    def _save_config(self, config):
        config_filename = f"{self.experiment_name}_config.yaml"
        yaml.dump(
            config,
            open(self.save_dir / config_filename, "w"),
        )
        self.logger.info(
            f"SAVED: YAML config file [{config_filename}] to [{self.save_dir}].\n"
        )

    def _block_by_shots(self) -> dict:
        """Create a group of frames for each shot.

        There should be 92 shots/blocks total.

        Returns:
            blocks_dict (dict): A dictionary of frames blocks.
        """
        shot_labels = self.video_data_loader.shot_scene_labels["shot_labels"]
        indices = np.unique(shot_labels)
        blocks_dict = {i: np.where(shot_labels == i)[0] for i in indices}
        self.logger.info(
            f"Frames blocked by shots. Each block is all the frames from a single shot."
        )
        return blocks_dict

    def _block_by_frames(self) -> dict:
        """Create groups of frames of size block_size for each shot.

        There should be 92 shots total.

        Returns:
            blocks_dict (dict): dictionary of frames blocks.
                The keys are the block indices, and the values are the indices of the frames in VideoDataLoader.frames.
        """
        shot_labels = self.video_data_loader.shot_scene_labels["shot_labels"]
        block_size = self.config["block_frames"]["block_size"]
        blocks = []
        for shot_idx in np.arange(len(np.unique(shot_labels))):
            shot_frames_idx = np.where(shot_labels == shot_idx)[0]
            shot_blocks = np.array(
                [
                    shot_frames_idx[i : i + block_size]
                    for i in range(0, len(shot_frames_idx), block_size)
                ],
                dtype=object,
            )
            blocks.extend(shot_blocks)
        blocks_dict = {i: block for i, block in enumerate(blocks)}
        self.logger.info(
            f"Frames grouped into blocks of size [{block_size}]. Each block only contains frames from "
            f"one shot."
        )
        return blocks_dict

    def _generate_splits(self, indices_to_split, y=None):
        """Generate the splits for KFold or StratifiedKFold.

        Args:
            indices_to_split (np.ndarray): indices for the data to split into folds
            y (np.ndarray): the class labels to use for class balanced StratifiedKFold splits

        Returns:

        """
        # Decide which class to use based on the 'kfolds_stratify' config
        KFoldClass = StratifiedKFold if self.config["kfolds_stratify"] else KFold

        # Create the fold generator with appropriate configuration
        fold_generator = KFoldClass(
            n_splits=self.config["n_folds"],
            shuffle=self.config["kfolds_shuffle"],
            random_state=self.config["seed"] if self.config["kfolds_shuffle"] else None,
        )

        # Generate and return the splits
        if self.config["kfolds_stratify"]:
            return fold_generator.split(indices_to_split, y)
        else:
            return fold_generator.split(indices_to_split)

    def _save_results(self, results, name="results", save_type="hdf5") -> None:
        if name == "":
            results_filename = f"{self.experiment_name}"
        else:
            results_filename = f"{self.experiment_name}_{name}"

        if save_type == "hdf5":
            results_filename += ".h5"
            results.to_hdf(self.save_dir / results_filename, key="results", mode="w")
        elif save_type == "pickle":
            results_filename += ".pkl"
            pickle.dump(results, open(self.save_dir / results_filename, "wb"))
        elif save_type == "csv":
            results_filename += ".csv"
            results.to_csv(self.save_dir / results_filename)
        else:
            raise ValueError("Save type not recognized.")

        self.logger.info(
            f"Results file [{results_filename}] saved to [{self.save_dir}]."
        )


if __name__ == "__main__":
    """
    For debugging purposes.
    """
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    neural_data_loader = NeuralDataLoader(config)
    video_data_loader = VideoDataLoader(config)
    experiment_runner = ExperimentRunner(
        config,
    )
