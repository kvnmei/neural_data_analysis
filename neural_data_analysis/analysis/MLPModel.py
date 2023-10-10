#!/usr/bin/env python3
import os
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml
from sklearn.metrics import r2_score

from neural_data_analysis.utils import correct_filepath


def plot_losses(losses, save_dir: Path, label=None, show_plot=False, run_info=None):
    """
    Plot the losses of a model.

    Args:
        losses (dict): dictionary with keys "train" and "val" and values that are lists of losses
        save_dir (str): directory to save the plot to
        label (str): label of the plot
        show_plot (bool): whether to show the plot or not

    Returns:
        None
    """
    _loss_names = ["train", "val"]
    save_dir = correct_filepath(f"{save_dir}/loss_plots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Plotting losses to {save_dir}...")
    _ = plt.figure()
    plt.suptitle(label)
    plt.scatter(
        np.arange(len(losses["train"])),
        losses["train"],
        marker=".",
        # label="train"
    )
    plt.scatter(
        np.arange(len(losses)),
        losses["val"],
        marker=".",
        # label="val"
    )
    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{label}_train_val_loss.png"))


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

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_mean_losses = []
        self.val_mean_losses = []

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.params["output_dims"]
        )
        self.train_acc = []
        self.val_acc = []
        self.train_mean_acc = []
        self.val_mean_acc = []

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
        # log the mean of the training losses
        self.train_mean_losses.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_mean_acc.append(torch.mean(torch.tensor(self.train_acc)))
        # reset the training losses after averaging them for the epoch
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
        # log the mean of the validation losses
        self.val_mean_losses.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_losses = []
        self.val_mean_acc.append(torch.mean(torch.tensor(self.val_acc)))
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


class MLPModel_Wrapper:
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

    # def create_dataset(self, X, y):
    #     dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    #     return dataset

    def create_dataloader(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # def create_train_val_dataloaders(self, X_train, y_train, X_val=None, y_val=None):
    #     train_dataset = self.create_dataset(X_train, y_train)
    #     train_dataloader = self.create_dataloader(train_dataset)
    #     if X_val is not None:
    #         val_dataset = self.create_dataset(X_val, y_val)
    #         val_dataloader = self.create_dataloader(val_dataset)
    #     else:
    #         val_dataloader = None
    #     return train_dataloader, val_dataloader

    def create_trainer(self):
        trainer = pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            num_sanity_val_steps=0,
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
            plot_losses(
                losses,
                run_info["save_dir"],
                filename=run_info_string,
                run_info=run_info,
            )

            # save self.model.train_mean_losses and self.model.val_mean_losses as csv using pandas
            csv_dir = correct_filepath(f"{run_info['save_dir']}/loss_values")
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            df = pd.DataFrame(
                {
                    "train_loss": np.array(self.model.train_mean_losses),
                    "val_loss": np.array(self.model.val_mean_losses),
                }
            )
            df.to_csv(os.path.join(csv_dir, f"{run_info_string}_losses.csv"))

            if hasattr(self.model, "train_mean_acc"):
                plots_dir = correct_filepath(f"{run_info['save_dir']}/acc_plots")
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                _ = plt.figure()
                plt.suptitle(run_info_string)
                plt.scatter(
                    np.arange(len(self.model.train_mean_acc)),
                    self.model.train_mean_acc,
                    marker=".",
                    # label="train"
                )
                plt.scatter(
                    np.arange(len(self.model.val_mean_acc)),
                    self.model.val_mean_acc,
                    marker=".",
                    # label="val"
                )
                plt.plot(self.model.train_mean_acc, label="train")
                plt.plot(self.model.val_mean_acc, label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{run_info_string}_train_val_acc.png")
                )

                # save self.model.train_mean_acc and self.model.val_mean_acc as csv using pandas
                csv_dir = correct_filepath(f"{run_info['save_dir']}/acc_values")
                if not os.path.exists(csv_dir):
                    os.makedirs(csv_dir)
                df = pd.DataFrame(
                    {
                        "train_acc": np.array(self.model.train_mean_acc),
                        "val_acc": np.array(self.model.val_mean_acc),
                    }
                )
                df.to_csv(os.path.join(csv_dir, f"{run_info_string}_acc.csv"))

    def predict(self, X):
        if not type(X) == torch.Tensor:
            X = torch.from_numpy(X).float()
        predictions = self.model(X)
        return predictions.detach().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)


class NeuralNetworkModel(object):
    """
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
            "hidden_dims": self.config["neuralNetwork"]["hidden_dims"],
            "num_layers": self.config["neuralNetwork"]["num_layers"],
            "learning_rate": self.config["neuralNetwork"]["lr"],
            "batch_size": self.config["neuralNetwork"]["batch_size"],
            "max_epochs": self.config["neuralNetwork"]["max_epochs"],
            "problem_type": self.config["neuralNetwork"]["problem_type"],
        }
        model = MLPModel_Wrapper(hparams)
        return model


if __name__ == "__main__":
    config = yaml.load(
        open("ExperimentRunner_config.yaml", "r"), Loader=yaml.FullLoader
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["experiment"]["backend"] == "pytorch":
        X_train = torch.rand(10000, 100)
        y_train = torch.rand(10000, 100)
        X_val = torch.rand(1000, 100)
        y_val = torch.rand(1000, 100)
        model = NeuralNetworkModel(config, device, X_train.shape[-1], y_train.shape[-1])
        model.fit(X_train, y_train, X_val, y_val)
        y_test = torch.rand(100, 100)
        predictions = model.predict(y_test)
        gt = torch.rand(100, 100)
        score = r2_score(gt, predictions)
        print(score)
