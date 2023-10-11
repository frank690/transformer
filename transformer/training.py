"""
This module contains the training related classes and functions.
"""

__all__ = ["train", "validate"]

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def save_model(
    model: nn.Module,
    path: str,
    epoch: int,
    optimizer: Optimizer,
    validation_loss: float,
) -> None:
    """
    This function saves the model to the given path.
    :param model: model to save
    :param path: path to save the model to
    :param epoch: current epoch
    :param optimizer: optimizer used for training
    :param validation_loss: current validation loss
    :return: None
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": validation_loss,
        },
        path,
    )


def load_model(model: nn.Module, path: str) -> nn.Module:
    """
    Loads the model from the given path.
    :param model: model to load
    :param path: path to load the model from
    :return: loaded model
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Model {path} is loaded from epoch {checkpoint['epoch']}, loss {checkpoint['loss']}"
    )
    return model


def train(
    model: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    criterion: _Loss,
    device: str,
) -> dict:
    """
    Trains the given model for the given number of epochs.
    :param model: model to train
    :param optimizer: optimizer to use
    :param num_epochs: number of epochs to train for
    :param training_dataloader: dataloader that provides training data
    :param validation_dataloader: dataloader that provides validation data
    :param criterion: loss function to use
    :param device: device in use
    :return: dictionary containing the training and validation loss
    """
    log = {"training_loss": [], "validation_loss": []}
    best_validation_loss = 1e8
    model = model.to(device)
    progress_bar = tqdm(range(num_epochs))

    for epoch in progress_bar:
        loss_current_epoch, training_accuracy = _train_one_epoch(
            model, optimizer, training_dataloader, criterion, device
        )
        validation_loss, validation_accuracy = _validate(
            model, validation_dataloader, criterion, device
        )

        message = f"Ep {epoch+1}/{num_epochs}: || Loss: Train {loss_current_epoch:.3f} \t Val {validation_loss:.3f}"
        progress_bar.set_description(message)

        log["training_loss"].append(loss_current_epoch)
        log["validation_loss"].append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_model(
                model,
                f"best_model_min_validation_loss.pth",
                epoch,
                optimizer,
                validation_loss,
            )
    return log


def _validate(
    model: nn.Module, validation_dataloader: DataLoader, criterion: _Loss, device: str
) -> tuple:
    """
    Validates the given model on the given validation data.
    :param model: model to validate
    :param validation_dataloader: dataloader that provides validation data
    :param criterion: loss function to use
    :param device: device in use
    :return: tuple of validation accuracy and validation loss
    """
    model.eval()
    correct, total = 0, 0
    loss_step = []
    with torch.no_grad():
        for data, labels in validation_dataloader:
            data, labels = data.to(device), labels.to(device)

            logits = model(data, labels)
            logits, labels = _prepare_logits_for_loss(logits=logits, labels=labels)

            validation_loss = criterion(logits, labels)

            predicted = torch.max(logits, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(validation_loss.item())

        validation_accuracy = (100 * correct / total).cpu().numpy().item()
        validation_loss_epoch = torch.tensor(loss_step).mean().numpy().item()
        return validation_loss_epoch, validation_accuracy


def _train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    training_dataloader: DataLoader,
    criterion: _Loss,
    device: str,
) -> tuple:
    """
    Trains the given model for one epoch.
    :param model: model to train
    :param optimizer: optimizer to use
    :param training_dataloader: dataloader that provides training data
    :param criterion: loss function to use
    :param device: device in use
    :return: tuple of loss and accuracy
    """
    model.train()
    loss_step = []
    correct, total = 0, 0
    for data, labels in training_dataloader:
        data, labels = data.to(device), labels.to(device)

        logits = model(data, labels)
        logits, labels = _prepare_logits_for_loss(logits=logits, labels=labels)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss_step.append(loss.item())

    loss_current_epoch = np.mean(loss_step)
    train_accuracy = (100 * correct / total).cpu().item()

    return loss_current_epoch, train_accuracy


def _prepare_logits_for_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple:
    """
    Prepares the logits and labels for the loss function.
    :param logits: logits (decoder output)
    :param labels: labels (true / expected output)
    :return: tuple of logits and labels
    """
    B, T, C = logits.shape
    return logits.view(B * T, C), labels.view(B * T)
