import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import numpy as np
from typing import Any


# Example plotting function referenced in your code.
def plot_metrics(
    metrics_dict: dict[str, Any], metric: str, save_dir: str, run_name: str
):
    fig, ax = plt.subplots(figsize=(6, 4))
    for phase, values in metrics_dict.items():
        ax.plot(values, label=f"{phase} {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{run_name} {metric.capitalize()}")
    ax.legend()
    fig.savefig(f"{save_dir}/{run_name}_{metric}.png", dpi=300)
    plt.close(fig)


class LSTMBinaryClassifier(pl.LightningModule):
    """
    LSTM-based binary classification model using PyTorch Lightning.
    """

    def __init__(self, hparams: dict[str, Any]):
        super().__init__()
        self.params = hparams
        input_dims = self.params["input_dims"]  # number of features per time step
        hidden_size = self.params[
            "hidden_dims"
        ]  # treat hidden_dims as LSTM hidden_size
        num_layers = self.params["num_layers"]
        output_dims = self.params["output_dims"]
        bidirectional: bool = self.params["bidirectional"]

        self.lstm = nn.LSTM(
            input_size=input_dims,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq, feature)
            dropout=0.0,  # adjust if needed
            bidirectional=bidirectional,
        )
        # hidden layer size is doubled if bidirectional
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_dims)

        # Set up loss and metrics
        pos_weight = self.params.get("pos_weights", None)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_losses = []
        self.val_losses = []
        self.train_mean_epoch_losses = []
        self.val_mean_epoch_losses = []

        self.accuracy = torchmetrics.classification.Accuracy(
            task="binary", num_classes=output_dims
        )
        self.train_acc = []
        self.val_acc = []
        self.train_mean_epoch_acc = []
        self.val_mean_epoch_acc = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dims)
        # LSTM returns output (batch_size, seq_len, hidden_size * num_directions) and (h_n, c_n)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        num_directions = 2 if self.params["bidirectional"] else 1

        # Extract the last layer's hidden states
        # h_n[-num_directions:] contains the last layer's forward and backward hidden states
        if self.params["bidirectional"]:
            # Concatenate the forward and backward hidden states
            last_hidden_forward = h_n[-2]  # Forward direction
            last_hidden_backward = h_n[-1]  # Backward direction
            last_hidden = torch.cat(
                (last_hidden_forward, last_hidden_backward), dim=1
            )  # (batch_size, hidden_size * 2)
        else:
            # h_n is the last hidden state of shape (num_layers, batch, hidden_size)
            # We take the last layer's hidden state for classification
            last_hidden = h_n[-1]  # shape (batch, hidden_size)
        logits = self.fc(last_hidden)  # shape (batch, output_dims)
        return logits

    def training_step(self, batch, batch_idx):
        (
            x,
            y,
        ) = batch  # x: [batch_size, seq_length, input_dims], y: [batch_size, output_dims]
        y_hat = self.forward(x)
        # y_hat: [batch_size, output_dims]
        # Ensure y is of type float and on the same device as y_hat
        y = y.float().to(y_hat.device)

        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.train_losses.append(loss.item())

        # use tensor operations to convert boolean values to float instead of Python's float() casting.
        preds = (torch.sigmoid(y_hat) > 0.5).float()
        acc = self.accuracy(preds, y)
        self.train_acc.append(acc.item())
        return loss

    def on_train_epoch_end(self):
        self.train_mean_epoch_losses.append(torch.mean(torch.tensor(self.train_losses)))
        self.train_mean_epoch_acc.append(torch.mean(torch.tensor(self.train_acc)))
        self.train_losses = []
        self.train_acc = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        y = y.float().to(y_hat.device)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_losses.append(loss.item())

        preds = (torch.sigmoid(y_hat) > 0.5).float()
        acc = self.accuracy(preds, y)
        self.val_acc.append(acc.item())
        return loss

    def on_validation_epoch_end(self):
        self.val_mean_epoch_losses.append(torch.mean(torch.tensor(self.val_losses)))
        self.val_mean_epoch_acc.append(torch.mean(torch.tensor(self.val_acc)))
        self.val_losses = []
        self.val_acc = []

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.params.get("learning_rate", 1e-3))
        return torch.optim.AdamW(
            self.parameters(), lr=self.params.get("learning_rate", 1e-3)
        )


class LSTMModelWrapper:
    """Wrapper for LSTMBinaryClassifier (and can be extended for other tasks).

    Methods:
        __init__(self, config: dict, input_dims: int, output_dims: int):
        create_dataloader(self, dataset):
        create_trainer(self, run_info: dict):
        fit(self, X_train, y_train, X_val=None, y_val=None, run_info=None):
        predict(self, X):
        save(self, path):
    """

    def __init__(self, config: dict[str, Any]):
        """
        Parameters:
            config (dict): dictionary with hyperparameters
            input_dims (int): number of features per time step
            output_dims (int): number of output classes (1 for binary)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hparams = {
            "input_dims": self.config["input_dims"],
            "output_dims": self.config["output_dims"],
            "hidden_dims": self.config["hidden_dims"],
            "num_layers": self.config["num_layers"],
            "learning_rate": self.config["lr"],
            "batch_size": self.config["batch_size"],
            "max_epochs": self.config["max_epochs"],
            "problem_type": self.config["problem_type"],
            "pos_weights": self.config.get("pos_weights", None),
            "bidirectional": self.config.get("bidirectional", False),
        }

        # Currently only binary classification is shown here, but you can add conditions
        # for multi_class or regression as needed.
        if self.hparams["problem_type"] == "binary_classification":
            self.model = LSTMBinaryClassifier(self.hparams)
        else:
            raise ValueError(
                f"Problem type {self.hparams['problem_type']} not supported for LSTM model. "
                f"Choose 'binary_classification' or implement a different class."
            )

    def create_dataloader(self, dataset: TensorDataset) -> DataLoader:
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=8,
            persistent_workers=True,
        )
        return dataloader

    def create_trainer(self, run_info: dict[str, Any]) -> pl.Trainer:
        csv_logger = CSVLogger(
            save_dir=run_info["save_dir"],
            name="lightning_logs",
        )
        trainer = pl.Trainer(
            max_epochs=self.hparams["max_epochs"],
            num_sanity_val_steps=0,
            accelerator="auto",
            devices=1,
            logger=csv_logger,
            log_every_n_steps=1,
        )
        return trainer

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        run_info: dict[str, Any] = None,
    ):
        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(X_train).float()
        if not torch.is_tensor(y_train):
            y_train = torch.from_numpy(y_train).float()
        dataset = TensorDataset(X_train, y_train)
        train_dataloader = self.create_dataloader(dataset)
        trainer = self.create_trainer(run_info)

        if X_val is not None and y_val is not None:
            if not torch.is_tensor(X_val):
                X_val = torch.from_numpy(X_val).float()
            if not torch.is_tensor(y_val):
                y_val = torch.from_numpy(y_val).float()
            dataset_val = TensorDataset(X_val, y_val)
            val_dataloader = DataLoader(
                dataset_val,
                batch_size=self.hparams["batch_size"],
                shuffle=False,
                drop_last=True,
                num_workers=8,
                persistent_workers=True,
            )
            trainer.fit(self.model, train_dataloader, val_dataloader)
        else:
            trainer.fit(self.model, train_dataloader)

        # plot losses if metrics are available
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

    def predict(self, X: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.detach().cpu().numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
