#!/usr/bin/env python3
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset


class MLPModel:
    """
     Multi-layer perceptron model.

    Methods:
        fit(self, X_train, y_train):
        predict(self, X_test):
        score(gt, predictions):

    """

    def __init__(
        self,
        config: dict,
        input_dims: int,
        output_dims: int,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_mlp(
            input_dims=input_dims,
            output_dims=output_dims,
        )

    # noinspection PyPep8Naming
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        run_info: dict = None,
    ):
        self.model.fit(X_train, y_train, X_val, y_val, run_info)

    # noinspection PyPep8Naming
    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def init_mlp(self, input_dims, output_dims):
        hparams = {
            "input_dims": input_dims,
            "output_dims": output_dims,
            "hidden_dims": self.config["ExperimentRunner"]["MLPModel"]["hidden_dims"],
            "num_layers": self.config["ExperimentRunner"]["MLPModel"]["num_layers"],
            "learning_rate": self.config["ExperimentRunner"]["MLPModel"]["lr"],
            "batch_size": self.config["ExperimentRunner"]["MLPModel"]["batch_size"],
            "max_epochs": self.config["ExperimentRunner"]["MLPModel"]["max_epochs"],
            "problem_type": self.config["ExperimentRunner"]["problem_type"],
        }
        model = MLPModelWrapper(hparams)
        return model


class MLPModelWrapper:
    """
    Wrapper for MLPClassifier and MLPRegressor.

    Methods:
        __init__(self, hparams):
        create_dataloader(self, dataset):
        create_trainer(self):
        fit(self, X_train, y_train, X_val=None, y_val=None, run_info=None):
        predict(self, X):
        save(self, path):

    """

    def __init__(self, hparams):
        """
        This model defines what type of MLP model to instantiate: classifier or regressor

        Args:
            hparams: hyperparameters
        """
        self.hparams = hparams
        if self.hparams["problem_type"] == "multi_class_classification":
            self.model = MLPMultiClassClassifier(hparams)
        elif self.hparams["problem_type"] == "binary_classification":
            self.model = MLPBinaryClassifier(hparams)
        elif self.hparams["problem_type"] == "regression":
            self.model = MLPRegressor(hparams)
        else:
            raise ValueError(
                f"Problem type {self.hparams['problem']} not supported. "
                f"Choose between 'classification' and 'regression'."
            )

    def create_dataloader(self, dataset: TensorDataset) -> DataLoader:
        # TODO: how to check how many workers to assign for num_workers arg?
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def create_trainer(self) -> pl.Trainer:
        # TODO: create a variable expected_epochs, and check the expected number of batches to see what value
        #  I should give for check_val_every_n_epochs, relates to log_every_n_steps too

        # TODO: distributed computing using two GPUs hangs... why?
        trainer = pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            num_sanity_val_steps=0,
            accelerator="auto",
            devices=1,
            # logger=None,
            # check_val_every_n_epoch=1,
        )
        return trainer

    # noinspection PyPep8Naming
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        run_info: dict = None,
    ):
        # convert to torch.Tensors if numpy arrays are provided
        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(X_train).float()
        if not torch.is_tensor(y_train):
            y_train = torch.from_numpy(y_train).float()
        dataset = TensorDataset(X_train, y_train)
        train_dataloader = self.create_dataloader(dataset)
        trainer = self.create_trainer()

        # if validation data is provided, use it
        if X_val is not None:
            if not torch.is_tensor(X_val):
                X_val = torch.from_numpy(X_val).float()
            if not torch.is_tensor(y_val):
                y_val = torch.from_numpy(y_val).float()
            dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
            val_dataloader = self.create_dataloader(dataset_val)
            trainer.fit(self.model, train_dataloader, val_dataloader)
        else:
            trainer.fit(self.model, train_dataloader)

        # plot losses
        if hasattr(self.model, "train_mean_epoch_losses"):
            losses = {
                "train": self.model.train_mean_epoch_losses,
                "val": self.model.val_mean_epoch_losses,
            }
            plot_metrics(
                losses,
                metric="loss",
                save_dir=run_info["save_dir"],
                run_name=run_info["run_name"],
            )

        # plot accuracies
        if hasattr(self.model, "train_mean_epoch_acc"):
            accuracies = {
                "train": self.model.train_mean_epoch_acc,
                "val": self.model.val_mean_epoch_acc,
            }
            plot_metrics(
                accuracies,
                metric="accuracy",
                save_dir=run_info["save_dir"],
                run_name=run_info["run_name"],
            )

    # noinspection PyPep8Naming
    def predict(self, X: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        predictions = self.model(X)
        return predictions.detach().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

class MLPBinaryClassifier(pl.LightningModule):
    """
    Methods:
        forward(self, x):
        training_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx):
        configure_optimizers(self):
    """

    def __init__(self, hparams):
        super(MLPBinaryClassifier, self).__init__()
        self.params = hparams
        num_layers = self.params["num_layers"]
        hidden_dims = self.params["hidden_dims"]

        # create using nn.Sequential
        self.network = torch.nn.Sequential()
        if num_layers == 1:
            self.network.add_module(
                "input",
                torch.nn.Linear(self.params["input_dims"], self.params["output_dims"]),
            )
        else:
            self.network.add_module(
                "input", torch.nn.Linear(self.params["input_dims"], hidden_dims)
            )
            self.network.add_module("batchnorm", torch.nn.BatchNorm1d(hidden_dims))
            self.network.add_module("relu", torch.nn.ReLU())
            for i in range(num_layers - 2):
                self.network.add_module(
                    "hidden" + str(i), torch.nn.Linear(hidden_dims, hidden_dims)
                )
                self.network.add_module(
                    "batchnorm" + str(i), torch.nn.BatchNorm1d(hidden_dims)
                )
                self.network.add_module("relu" + str(i), torch.nn.ReLU())
            self.network.add_module(
                "output", torch.nn.Linear(hidden_dims, self.params["output_dims"])
            )

        # cross entropy loss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_mean_epoch_losses = []
        self.val_mean_epoch_losses = []

        # classification accuracy
        self.accuracy = torchmetrics.classification.Accuracy(
            task="binary", num_classes=self.params["output_dims"]
        )
        self.train_acc = []
        self.val_acc = []
        self.train_mean_epoch_acc = []
        self.val_mean_epoch_acc = []

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.train_losses.append(loss.item())
        predictions = (torch.sigmoid(y_hat) > 0.5).float()
        acc = self.accuracy(predictions, y)
        self.train_acc.append(acc.item())
        return loss

    def on_train_epoch_end(self) -> None:
        # log the mean of the training metrics
        self.train_mean_epoch_losses.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_mean_epoch_acc.append(torch.mean(torch.tensor(self.train_acc)))
        # reset the training metrics after averaging them for the epoch
        self.train_losses = []
        self.train_acc = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_losses.append(loss.item())
        predictions = (torch.sigmoid(y_hat) > 0.5).float()
        acc = self.accuracy(predictions, y)
        self.val_acc.append(acc.item())
        return loss

    def on_validation_epoch_end(self) -> None:
        # log the mean of the validation metrics
        self.val_mean_epoch_losses.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_mean_epoch_acc.append(torch.mean(torch.tensor(self.val_acc)))
        # reset the validation metrics after averaging them for the epoch
        self.val_losses = []
        self.val_acc = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameters(), lr=self.params["learning_rate"]
        )


class MLPMultiClassClassifier(pl.LightningModule):
    """
    Methods:
        forward(self, x):
        training_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx):
        configure_optimizers(self):
    """

    def __init__(self, hparams):
        super(MLPMultiClassClassifier, self).__init__()
        self.params = hparams
        num_layers = self.params["num_layers"]
        hidden_dims = self.params["hidden_dims"]

        # create using nn.Sequential
        self.network = torch.nn.Sequential()
        if num_layers == 1:
            self.network.add_module(
                "input",
                torch.nn.Linear(self.params["input_dims"], self.params["output_dims"]),
            )
        else:
            self.network.add_module(
                "input", torch.nn.Linear(self.params["input_dims"], hidden_dims)
            )
            self.network.add_module("batchnorm", torch.nn.BatchNorm1d(hidden_dims))
            self.network.add_module("relu", torch.nn.ReLU())
            for i in range(num_layers - 2):
                self.network.add_module(
                    "hidden" + str(i), torch.nn.Linear(hidden_dims, hidden_dims)
                )
                self.network.add_module(
                    "batchnorm" + str(i), torch.nn.BatchNorm1d(hidden_dims)
                )
                self.network.add_module("relu" + str(i), torch.nn.ReLU())
            self.network.add_module(
                "output", torch.nn.Linear(hidden_dims, self.params["output_dims"])
            )
        self.network.add_module("softmax", torch.nn.Softmax(dim=1))

        # cross entropy loss
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_mean_epoch_losses = []
        self.val_mean_epoch_losses = []

        # classification accuracy
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.params["output_dims"]
        )
        self.train_acc = []
        self.val_acc = []
        self.train_mean_epoch_acc = []
        self.val_mean_epoch_acc = []

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y.long())
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.train_losses.append(loss.item())
        predictions = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(predictions, y)
        self.train_acc.append(acc.item())
        return loss

    def on_train_epoch_end(self) -> None:
        # log the mean of the training metrics
        self.train_mean_epoch_losses.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_mean_epoch_acc.append(torch.mean(torch.tensor(self.train_acc)))
        # reset the training metrics after averaging them for the epoch
        self.train_losses = []
        self.train_acc = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y.long())
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_losses.append(loss.item())
        predictions = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(predictions, y)
        self.val_acc.append(acc.item())
        return loss

    def on_validation_epoch_end(self) -> None:
        # log the mean of the validation metrics
        self.val_mean_epoch_losses.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_mean_epoch_acc.append(torch.mean(torch.tensor(self.val_acc)))
        # reset the validation metrics after averaging them for the epoch
        self.val_losses = []
        self.val_acc = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameters(), lr=self.params["learning_rate"]
        )


class MLPRegressor(pl.LightningModule):
    """
    Methods:
        forward(self, x):
        training_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx):
        configure_optimizers(self):
    """

    def __init__(self, hparams):
        super(MLPRegressor, self).__init__()
        self.params = hparams
        num_layers = self.params["num_layers"]
        hidden_dims = self.params["hidden_dims"]

        # create using nn.Sequential
        self.network = torch.nn.Sequential()
        if num_layers == 1:
            self.network.add_module(
                "input",
                torch.nn.Linear(self.params["input_dims"], self.params["output_dims"]),
            )
        else:
            self.network.add_module(
                "input", torch.nn.Linear(self.params["input_dims"], hidden_dims)
            )
            self.network.add_module("batchnorm", torch.nn.BatchNorm1d(hidden_dims))
            self.network.add_module("relu", torch.nn.ReLU())
            for i in range(num_layers - 2):
                self.network.add_module(
                    "hidden" + str(i), torch.nn.Linear(hidden_dims, hidden_dims)
                )
                self.network.add_module(
                    "batchnorm" + str(i), torch.nn.BatchNorm1d(hidden_dims)
                )
                self.network.add_module("relu" + str(i), torch.nn.ReLU())
            self.network.add_module(
                "output", torch.nn.Linear(hidden_dims, self.params["output_dims"])
            )

        self.criterion = torch.nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.train_mean_losses = []
        self.val_mean_losses = []

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self) -> None:
        # log the mean of the training losses
        self.train_mean_losses.append(torch.mean(torch.tensor(self.train_losses)))
        # reset the training losses after averaging them for the epoch
        self.train_losses = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.network(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_losses.append(loss.item())
        return loss

    def on_validation_epoch_end(self) -> None:
        # log the mean of the validation losses
        self.val_mean_losses.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_losses = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameters(), lr=self.params["learning_rate"]
        )


def plot_metrics(
    values: dict,
    metric: str = "losses",
    save_dir: Path = Path("plots"),
    run_name: str = "",
) -> None:
    """
    Plot the losses of a model.

    Args:
        values (dict): dictionary with keys "train" and "val" and values that are lists of values
        metric (str): metric to plot
        save_dir (Path): directory to save plots and csv files for metric values
        run_name (str): name of the run

    Returns:
        None
    """
    _loss_names = ["train", "val"]
    plots_dir = Path(f"{save_dir}/{metric}_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(f"{save_dir}/{metric}_values")
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {metric} to {save_dir}...")
    _ = plt.figure()
    plt.suptitle(run_name)
    plt.scatter(
        np.arange(len(values["train"])),
        values["train"],
        marker=".",
        # label="train"
    )
    plt.scatter(
        np.arange(len(values["val"])),
        values["val"],
        marker=".",
        # label="val"
    )
    plt.plot(values["train"], label="train")
    plt.plot(values["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{run_name}_train_val_{metric}.png")

    values_df = pd.DataFrame(values)
    values_df.to_csv(csv_dir / f"{run_name}_{metric}.csv")
