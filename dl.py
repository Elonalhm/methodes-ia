"""Functions for the deep learning mode.

Notes
-----
Inspired from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html.
"""

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from viz import training_curves

def get_FashionMNIST_datasets(batch_size=64, only_loader=True):
    """Loads and returns the FashionMNIST dataset.

    The data is returned as DataLoader objects by default or,
    if specified by the user, also as datasets.

    Parameters
    ----------
    batch_size: int, default=64
        The batch size to use for the DataLoader objects.
    only_loader: bool, default=True
        If `True`, returns only the DataLoader objects.
        If `False`, also returns the datasets.

    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    training_data: torchvision.datasets.mnist.FashionMNIST
        The training data, only returned if `only_loader==False`.
    test_data: torchvision.datasets.mnist.FashionMNIST
        The test data, only returned if `only_loader==False`.

    """
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if only_loader:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, training_data, test_data


class BatchNormalization:
    """ A changer : Normalizes the input data for each of its activations, then scale and shift them
    
    Parameters
    ----------
    input_size: int
        Number of neurons in the layer
    eps: float, default=1e-5
        The small constant added for stability.
    momentum: float
        Forgetting factor for moving averages and moving variances.
    learning_rate: float, default=0.0
        Multiplicative factor applied to the gradient to vary the gain of the gradient.
    
    """
    def __init__(self, input_size, eps=1e-5, momentum=0.9, learning_rate=0.0):
        self.eps = eps
        self.learning_rate = learning_rate
        self.momentum = momentum 
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
        
    def forward_BN(self, x, train=True):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.
        train: Boolean, default=True
            True, if we are not in inference mode 
            False, if we do

        Returns
        -------
        The scaled and shifted normalized activations of the input and the normalized input.
        """
        if train:
            # Calculate the mean and variance from the mini-batch
            m=len(x)
            mu = (1/m)*np.sum(x, axis=0)
            var = (1/m)*np.sum((x-mu)**2, axis=0)
            # Update moving averages and moving variances
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # If we are in inference mode, use moving averages and stored moving variances
            mu = self.running_mean
            var = self.running_var
        
        # Normalization
        x_norm = (x-mu)/np.sqrt(var + self.eps)
        
        # Scaling and shifting
        out = self.gamma * x_norm + self.beta
        
        return out, x_norm
    
    def backward_BN(self, dout, x_norm):
        """The backward pass.

        Parameters
        ----------
        dout : Tensor
            Gradient of the input of the forward function.
        x_norm: Tensor
            The normalized input.
        Returns
        -------
        The scaled and shifted normalized activations of the input and the normalized input.
        """
        # Calculate the gradients of gamma, beta and x_norm
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_norm = dout * self.gamma
        
        # Calculate the mu and var gradients
        N = x_norm.shape[0]
        dvar = np.sum(dx_norm * (self.x - self.running_mean) * (-0.5) * (self.var + self.eps)**(-3/2), axis=0)
        dmu = np.sum(dx_norm * (-1 / np.sqrt(self.var + self.eps)), axis=0) + dvar * np.mean(-2 * (self.x - self.running_mean), axis=0)
        
        # Calculate the gradient of the input
        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.running_mean) / N + dmu / N
        
        # Update the parameters of the Batch Normalization
        self.gamma -= self.learning_rate * dgamma
        self.beta -= self.learning_rate * dbeta
        
        return dx

class LayerNormalization:
    """ A changer : Normalizes the input data for each of its activations, then scale and shift them
    
    Parameters
    ----------
    input_size: int
        Number of neurons in the layer
    eps: float, default=1e-5
        The small constant added for stability.
    alpha: float, default=0.99
        Controls normalization. 
    learning_rate: float, default=0.0
        Multiplicative factor applied to the gradient to vary the gain of the gradient.
    
    """
    def __init__(self, input_size, eps=1e-5, alpha=0.99, learning_rate=0.0):
        self.eps = eps 
        self.alpha = alpha 
        self.learning_rate=learning_rate
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.running_mean = np.zeros(input_size)
        self.running_var = np.ones(input_size)
        
    def forward_LN(self, x, train=True):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.
        train: Boolean, default=True
            True, if we are not in inference mode 
            False, if we do

        Returns
        -------
        The scaled and shifted normalized activations of the input and the normalized input.
        """
        if train:
            # Calculate the mean and variance from all layer activations
            mu = np.mean(x, axis=1, keepdims=True)
            var = np.var(x, axis=1, keepdims=True)
            # Update moving average and moving variance
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mu
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * var
        else:
            # If we are in inference mode, use the moving average and the moving variance stored
            mu = self.running_mean
            var = self.running_var
        
        # Normalization
        x_norm = (x - mu) / np.sqrt(var + self.eps)
        
        # Scaling and shifting
        out = self.gamma * x_norm + self.beta
        
        return out, x_norm
    
    def backward_LN(self, dout, x_norm):
        """The backward pass.

        Parameters
        ----------
        dout : Tensor
            Gradient of the input of the forward function.
        x_norm: Tensor
            The normalized input.
         
        Returns
        -------
        The scaled and shifted normalized activations of the input and the normalized input.
        """
        # Calculate the gradients of gamma, beta and x_norm
        dgamma = np.sum(dout * x_norm, axis=1, keepdims=True)
        dbeta = np.sum(dout, axis=1, keepdims=True)
        dx_norm = dout * self.gamma
        
        # Calculate the mu and var gradients
        N = x_norm.shape[1]
        dvar = np.sum(dx_norm * (self.x - self.running_mean) * (-0.5) * (self.var + self.eps)**(-3/2), axis=1, keepdims=True)
        dmu = np.sum(dx_norm * (-1 / np.sqrt(self.var + self.eps)), axis=1, keepdims=True) + dvar * np.mean(-2 * (self.x - self.running_mean), axis=1, keepdims=True)
        
        # Calculate the gradient of the input
        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.running_mean) / N + dmu / N
        
        # Update Layer Normalization settings
        self.gamma -= self.learning_rate * np.mean(dgamma, axis=0)
        self.beta -= self.learning_rate * np.mean(dbeta, axis=0)
        
        return dx

# Define model
class FMNIST_MLP(nn.Module):
    """The MLP model we train on the FashionMNIST dataset.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Attributes
    ----------
    flatten: nn.Flatten
        A flatten layer.
    linear_relu_stack: nn.Sequential
        A stack of liner layers with ReLU
        activations.
    metrics: pd.DataFrame
        The training metrics dataframe.
    """

    def __init__(self, hidden_layers=2, dropout_rate=0.0):

        super().__init__()
        self.flatten = nn.Flatten()
        self.list_hidden = []
        for _ in range(hidden_layers - 1):
            #step 4(merci Elona) :ajouter ici via BNNetwork: list_hidden.append(lambda x : BNT(x))
            self.list_hidden.append(nn.Linear(512, 512))
            self.list_hidden.append(nn.ReLU())
            #(juste pour comprendre) à la place de nn.ReLu() on pourrait mettre : list_hidden.append(lambda x : np.max(x, 0))
            self.list_hidden.append(nn.Dropout(dropout_rate))
        self.linear_relu_stack = nn.Sequential(
            #step 3 : BNT aussi
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *self.list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        logits: Tensor
            The unnormalized logits, of shape `(batch_size, 10)`.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        """Sets the metrics dataframe.

        Used when a saved model is loaded,
        to also load its past training metrics dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The training metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = df

    def update_metrics(self, series):
        """Updates the metrics dataframe after one epoch.

        Parameters
        ----------
        series: pd.Series
            The new row of metrics to add at the
            end of the metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


class BN_FMNIST_MLP(nn.Module):
    """The MLP model we train on the FashionMNIST dataset.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Attributes
    ----------
    flatten: nn.Flatten
        A flatten layer.
    linear_relu_stack: nn.Sequential
        A stack of liner layers with ReLU
        activations.
    metrics: pd.DataFrame
        The training metrics dataframe.
    """

    def __init__(self, hidden_layers=2, dropout_rate=0.0):

        super().__init__()
        self.flatten = nn.Flatten()
        self.list_hidden = []
        for _ in range(hidden_layers - 1):
            self.list_hidden.append(nn.Linear(512, 512))
            self.list_hidden.append(self.BatchNormalization(512))
            self.list_hidden.append(nn.ReLU())
            
            self.list_hidden.append(nn.Dropout(dropout_rate))
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(28 * 28, 512),
            self.BatchNormalization(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *self.list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        logits: Tensor
            The unnormalized logits, of shape `(batch_size, 10)`.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        """Sets the metrics dataframe.

        Used when a saved model is loaded,
        to also load its past training metrics dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The training metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = df

    def update_metrics(self, series):
        """Updates the metrics dataframe after one epoch.

        Parameters
        ----------
        series: pd.Series
            The new row of metrics to add at the
            end of the metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)


class LN_FMNIST_MLP(nn.Module):
    """The MLP model we train on the FashionMNIST dataset.

    Parameters
    ----------
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.

    Attributes
    ----------
    flatten: nn.Flatten
        A flatten layer.
    linear_relu_stack: nn.Sequential
        A stack of liner layers with ReLU
        activations.
    metrics: pd.DataFrame
        The training metrics dataframe.
    """

    def __init__(self, hidden_layers=2, dropout_rate=0.0):

        super().__init__()
        self.flatten = nn.Flatten()
        self.list_hidden = []
        for _ in range(hidden_layers - 1):
            self.list_hidden.append(nn.Linear(512, 512))
            self.list_hidden.append(self.LayerNormalization(512))
            self.list_hidden.append(nn.ReLU())
            
            self.list_hidden.append(nn.Dropout(dropout_rate))
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(28 * 28, 512),
            self.LayerNormalization(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *self.list_hidden,
            nn.Linear(512, 10),
        )
        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    def forward(self, x):
        """The forward pass.

        Parameters
        ----------
        x: Tensor
            The input tensor, of shape `(batch_size, 1, 28, 28)`.

        Returns
        -------
        logits: Tensor
            The unnormalized logits, of shape `(batch_size, 10)`.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def set_metrics(self, df):
        """Sets the metrics dataframe.

        Used when a saved model is loaded,
        to also load its past training metrics dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            The training metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = df

    def update_metrics(self, series):
        """Updates the metrics dataframe after one epoch.

        Parameters
        ----------
        series: pd.Series
            The new row of metrics to add at the
            end of the metrics dataframe.

        Returns
        -------
        None

        """
        self.metrics = pd.concat([self.metrics, series.to_frame().T], ignore_index=True)

def train(dataloader, model, loss_fn, optimizer, device, mode=None):
    """The training step for one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The training DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    optimizer: torch.optim.optimizer.Optimizer
        The optimizer.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    train_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on the epoch,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if mode == "script":
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test(dataloader, model, loss_fn, device, mode=None):
    """The evaluation step after one epoch.

    Arguments
    ---------
    dataloader: torch.utils.data.DataLoader
        The test DataLoader.
    model: nn.Module
        The model.
    loss_fn: nn.modules._Loss
        The loss function.
    device: str
        The device to use, `"gpu"` or `"cpu"`.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    test_loss: float
        The averaged loss on all the batches,
        which will be added to the metrics dataframe.
    correct: float
        The accuracy of all the predictions on all the batches,
        which will be added to the metrics dataframe.

    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if mode == "script":
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return test_loss, correct


def get_and_train_model(
    train_dataloader,
    test_dataloader,
    hidden_layers=2,
    dropout_rate=0.0,
    epochs=5,
    mode=None,
):
    """Creates and trains a model on the given dataset.

    Doesn't train if saved weights are found for the given hyperparameters
    (except the number of epochs). The MLP architecture is displayed.

    Parameters
    ----------
    train_dataloader: torch.utils.data.DataLoader
        The DataLoader of the training data.
    test_dataloader: torch.utils.data.DataLoader
        The DataLoader of the test data.
    hidden_layers: int, default=2
        The number of hidden fully connected layers.
    dropout_rate: float, default=0
        The dropout rate.
    epochs: int, default=5
        The number of epochs used for training.
    mode: str
        Either `"script"` if the module is used as a script,
        or `"st"` if used in the stramlit app. This governs
        the kind of outputs produced (prints, figures).

    Returns
    -------
    model: FMNIST_MLP
        The model.
    """
    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "script":
        print(f"Using {device} device")

    # Create the model
    model = FMNIST_MLP(hidden_layers, dropout_rate)
    base_name = (
        "saved_models/fmnist_mlp_hidden="
        + str(hidden_layers)
        + "_dropout_rate="
        + str(dropout_rate)
    )
    path = base_name + ".pth"
    path_metrics = base_name + "_metrics.csv"

    # Load the weights if they already exist
    if os.path.exists(path):
        if mode == "script":
            print("model already exists, let us just load it")
        elif mode == "st":
            st.write("Found a saved model with given config")
        model.load_state_dict(torch.load(path))
        metrics = pd.read_csv(path_metrics, index_col=0)
        model.set_metrics(metrics)
    model = model.to(device)
    if mode == "script":
        print(model)
    elif mode == "st":
        st.text("Model architecture:")
        st.text(model)

    # Train the model and save the wieghts if they don't exist
    if not os.path.exists(path):
        if mode == "script":
            print("no existing model found")
        elif mode == "st":
            st.write("Didn't find an existing model, training a new one")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for t in range(epochs):
            if mode == "script":
                print(f"Epoch {t+1}\n-------------------------------")
            train_loss, train_acc = train(
                train_dataloader, model, loss_fn, optimizer, device, mode
            )
            test_loss, test_acc = test(test_dataloader, model, loss_fn, device, mode)

            # Saved the metrics in the model.metrics dataframe
            new_row = pd.Series(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            model.update_metrics(new_row)
            if (mode == "st") & ((t + 1) % 10 == 0):
                st.text(
                    f"End of epoch {t+1}, Test Error:\n Accuracy: {(100 * new_row['test_acc']):>0.1f}%, Avg loss: {new_row['test_loss']:>8f}"
                )

        if mode == "script":
            print("Done!")

        # Save the weights and the metrics dataframe
        torch.save(model.state_dict(), path)
        model.metrics.to_csv(path_metrics)
        if mode == "script":
            print("Saved PyTorch Model State to " + path)
    if mode == "script":
        print(model.metrics)
    return model


if __name__ == "__main__":

    mode = "script"

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    args = parser.parse_args()

    train_dataloader, test_dataloader = get_FashionMNIST_datasets(64)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = get_and_train_model(
        train_dataloader,
        test_dataloader,
        hidden_layers=args.hidden,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        mode=mode,
    )
    training_curves(model, mode)
