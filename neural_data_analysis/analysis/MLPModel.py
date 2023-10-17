#!/usr/bin/env python3
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics


class MLPModel(object):
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
        device: torch.device,
        input_dims: list,
        output_dims: list,
    ):
        self.config = config
        self.device = device
        self.model = self.init_mlp(
            device,
            input_dims=input_dims,
            output_dims=output_dims,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, run_info=None):
        self.model.fit(X_train, y_train, X_val, y_val, run_info)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def init_mlp(self, device, input_dims, output_dims):
        hparams = {
            "input_dims": input_dims,
            "output_dims": output_dims,
            "hidden_dims": self.config["NeuralNetwork"]["hidden_dims"],
            "num_layers": self.config["NeuralNetwork"]["num_layers"],
            "learning_rate": self.config["NeuralNetwork"]["lr"],
            "batch_size": self.config["NeuralNetwork"]["batch_size"],
            "max_epochs": self.config["NeuralNetwork"]["max_epochs"],
            "problem_type": self.config["NeuralNetwork"]["problem_type"],
        }
        model = MLPModel_Wrapper(hparams)
        return model


class MLPModel_Wrapper:
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
        self.hparams = hparams
        if self.hparams["problem_type"] == "classification":
            self.model = MLPClassifier(hparams)
        elif self.hparams["problem_type"] == "regression":
            self.model = MLPRegressor(hparams)
        else:
            raise ValueError(
                f"Problem type {self.hparams['problem']} not supported. "
                f"Choose between 'classification' and 'regression'."
            )

    def create_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def create_trainer(self):
        trainer = pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            num_sanity_val_steps=0,
            # logger=None,
            # check_val_every_n_epoch=1,
        )
        return trainer

    def fit(self, X_train, y_train, X_val=None, y_val=None, run_info=None):
        val_dataloader = None
        if not type(X_train) == torch.Tensor:
            X_train = torch.from_numpy(X_train).float()
        if not type(y_train) == torch.Tensor:
            y_train = torch.from_numpy(y_train).float()
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_dataloader = self.create_dataloader(dataset)
        trainer = self.create_trainer()

        # if validation data is provided, use it
        if X_val is not None:
            if not type(X_val) == torch.Tensor:
                X_val = torch.from_numpy(X_val).float()
            if not type(y_val) == torch.Tensor:
                y_val = torch.from_numpy(y_val).float()
            dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
            val_dataloader = self.create_dataloader(dataset_val)
            trainer.fit(self.model, train_dataloader, val_dataloader)
        else:
            trainer.fit(self.model, train_dataloader)

        # plot losses
        run_info_string = (
            f"{run_info['brain_area']}"
            f"_{run_info['bin_center']}"
            f"_{run_info['bin_size']}"
            f"_{run_info['embedding']}"
            f"_fold{run_info['fold']}"
        )
        losses = {
            "train": self.model.train_losses,
            "val": self.model.val_losses,
        }
        plot_metrics(
            losses,
            metric="loss",
            save_dir=run_info["save_dir"],
            filename=run_info_string,
            run_info=run_info,
        )

        # plot accuracies
        if hasattr(self.model, "train_mean_acc"):
            accuracies = {
                "train": self.model.train_losses,
                "val": self.model.val_losses,
            }



    def predict(self, X):
        if not type(X) == torch.Tensor:
            X = torch.from_numpy(X).float()
        predictions = self.model(X)
        return predictions.detach().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)



class MLPClassifier(pl.LightningModule):
    """
    Methods:
        forward(self, x):
        training_step(self, batch, batch_idx):
        validation_step(self, batch, batch_idx):
        configure_optimizers(self):
    """

    def __init__(self, hparams):
        super(MLPClassifier, self).__init__()
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



def plot_metrics(values: dict, metric: str = "losses", save_dir: Path = Path("plots"), label=None, show_plot=False, run_info=None) -> None:
    """
    Plot the losses of a model.

    Args:
        values (dict): dictionary with keys "train" and "val" and values that are lists of values
        metric (str): metric to plot
        save_dir (Path): directory to save plots and csv files for metric values
        label (str): label of the plot
        show_plot (bool): whether to show the plot or not

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
    plt.suptitle(label)
    plt.scatter(
        np.arange(len(values["train"])),
        values["train"],
        marker=".",
        # label="train"
    )
    plt.scatter(
        np.arange(len(values)),
        values["val"],
        marker=".",
        # label="val"
    )
    plt.plot(values["train"], label="train")
    plt.plot(values["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{label}_train_val_{metric}.png")

    values_df = pd.DataFrame(values)
    values_df.to_csv(csv_dir / f"{label}_{metric}.csv")