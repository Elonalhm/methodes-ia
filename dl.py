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

def mean(batch):
    """gives the mean of the input
    
    Parameters
    ----------
    batch: list
        The input data
    
    Returns
    ----------
    The mean of the data
    """
    m=len(batch)
    return (1/m)*np.sum(batch, axis=0)

def variance(batch, moy = None):
    """gives the variance of the input
    
    Parameters
    ----------
    batch: list
        The input data
    moy: type of elements in batch, default=None
        the mean of the batch if it has already been calculated
    
    Returns
    ----------
    The Variance of the data
    """
    m=len(batch)
    if moy is None:
        moy = mean(batch)
    return (1/m)*np.sum((batch-moy)**2, axis=0)

def BNT(batch, gamma, beta, eps=0.1):
    """Normalizes the input data for each of its activations, then scale and shift them
    
    Parameters
    ----------
    batch: list
        The non-normalized input mini-batch of data.
    gamma: float
        The scale parameter.
    beta: float
        The shift parameter.
    eps: float, default=0.1
        The small constant added for stability.
    

    Returns
    ----------
    The scaled and shifted normalized activations of the input
    
    """

    mu_b = mean(batch)
    sigma2 = variance(batch, mu_b)
    x_chap = (batch-mu_b)/np.sqrt(sigma2+eps)
    return gamma*x_chap+beta

def append_list(liste, *args):
    """Appends args at the end of the list, modifies the input list
    
    Parameters
    ----------
    liste: list
        A list.
    args: Any
        added objects.

    Modifies 'liste'
    """
    for x in args:
        liste.append(x)

def insert_BNT(liste, element, criteria, before):
    """returns a list, based on 'liste' in which objects are added following a criteria, each time the criteria is completed
    
    Parameters
    ----------
    liste: list
        The input list.
    element: Any
        The inserted element.
    criteria: fun : element of 'liste' -> boolean
        When it is inserted.
    before: boolean
        Where it is inserted : if True, before the element, else after
    
    Returns
    -----------
    new : 'liste' with 'element' inserted based on the condition 'criteria' and 'before'
    """
    new = []
    for x in liste:
        if criteria(x):
            if before:
                append_list(new, element, x)
            else:
                append_list(new, x, element)
        else:
            new.append(x)
    return new

def replace_BNT(liste, new, condition):
    """Replaces elements of liste by new ones based on a condition, modifies the input list

    Parameters
    ----------

    liste: list
        The input list
    new: Any
        The replacing object
    condition: fun : element of list -> boolean
        The condition on which the element is replaced
    
    Modifies 'liste'
    """
    for i, x in enumerate(liste):
        if condition(x):
            liste[i] = new
    return liste




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
        self.input = [
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            ]
        self.list_hidden = []
        self.output = [nn.Linear(512, 10)]
        
        for _ in range(hidden_layers - 1):
            #step 4(merci Elona) :ajouter ici via BNNetwork: list_hidden.append(lambda x : BNT(x))
            self.list_hidden.append(nn.Linear(512, 512))
            self.list_hidden.append(nn.ReLU())
            #(juste pour comprendre) à la place de nn.ReLu() on pourrait mettre : list_hidden.append(lambda x : np.max(x, 0))
            self.list_hidden.append(nn.Dropout(dropout_rate))
        
        self.linear_relu_stack = nn.Sequential(
            #step 3 : BNT aussi
            *self.input,
            *self.list_hidden,
            *self.output
        )

        self.metrics = pd.DataFrame(
            columns=["train_loss", "train_acc", "test_loss", "test_acc"]
        )

    """getters"""
    def get_flatten(self):
        return self.flatten
        
    def get_input(self):
        return self.input
    
    def get_list_hidden(self):
        return self.list_hidden
    
    def get_output(self):
        return self.output
    
    def get_linear_relu_stack(self):
        return self.linear_relu_stack
    
    def get_metrics(self):
        return self.metrics
    

    """setters"""
    def set_flatten(self, f):
        self.flatten = f
        
    def set_input(self, i):
        self.input = i
    
    def set_list_hidden(self, liste):
        self.list_hidden = liste
    
    def set_output(self, o):
        self.output = o
    
    def set_linear_relu_stack(self, lrs):
        self.linear_relu_stack = lrs

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


def BNNetwork(N, subset, optimizer, loss_fn, gamma = 1, beta = 0, before_ReLU = True):
    """Creates a Batch Normalised Network from a basic network with optimized hyperparameters
    
    Parameters
    ----------
    N: FMNIST_MLP
        a MLP network.
    subset: list
        a batch?
    gamma: float, default=1
        The scale parameter to be optimized.
    beta: float, default=0
        The shift parameter to be optimized.
    before_ReLU: boolean, default=True
        Applies the transformation before the activation function
        if True, else right after it
    optimizer : string
        Minimise the loss function
    loss_fn : string
        The loss function

    Returns
    ----------
    A batch normalised network with fixed, optimized hyperparameters.
    
    """

    N_tr_BN = N.copy()

    BN = lambda x : BNT(x, gamma, beta)
    input = N_tr_BN.get_input()
    list_hidden = N_tr_BN.get_list_hidden()
    output = N_tr_BN.get_output()
    

    #step 3
    new_input = insert_BNT(input, BN, lambda x : isinstance(x, torch.nn.modules.activation.ReLU), before_ReLU)
    #step 4
    new_hidden = insert_BNT(list_hidden, BN, lambda x : isinstance(x, torch.nn.modules.activation.ReLU), before_ReLU)
    
    N_tr_BN.set_list_hidden(new_hidden)
    N_tr_BN.set_input(new_input)
    N_tr_BN.set_linear_relu_stack = nn.Sequential(
            *input,
            *list_hidden,
            *output
        )


    #step 6 optimiser les hyper sur le réseau TODO
    
    gamma = None
    beta = None

    #step 7
    N_inf_BN = N_tr_BN

    #step 10???????? TODO
    wtf = "wtf"
    esp = mean(wtf)
    var = variance(wtf, esp)

    #step 11
    #wtf is epsilon TODO
    epsilon = "what the f*** am I"
    temp = gamma/np.sqrt(var+epsilon)
    new_BN = lambda x : (temp)*x + (beta - temp*esp)
    
    N_inf_BN.set_input(replace_BNT(N_inf_BN.get_input(), new_BN, lambda x : x.__name__ == "<lambda>"))
    N_inf_BN.set_list_hidden(replace_BNT(N_inf_BN.get_list_hidden(), new_BN, lambda x : x.__name__ == "<lambda>"))
    N_inf_BN.set_linear_relu_stack = nn.Sequential(
            *input,
            *list_hidden,
            *output
        )



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
    training_curves(model, "st")
