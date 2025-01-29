"""
Christopher Ondrusz
GitHub: acse_cro23
"""
from .utils import loss_function, surrogate_target, compute_alpha_sigma, noise_schedule # noqa
import torch.nn as nn
import torch
from torch.optim import Adam


class PredictionModel(nn.Module):
    def __init__(self,
                 model,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        A wrapper class for a neural network model that facilitates training,
        validation, and saving/loading model checkpoints. The model is designed
        to run on either a CPU or GPU.

        Parameters:
        -----------
        model : nn.Module
            The base model architecture (e.g., UNet).
        device : str, optional, default='cuda' if torch.cuda.is_available() else 'cpu' # noqa
            The device to run the model on ('cuda' or 'cpu').

        Returns:
        --------
        None
        """
        super(PredictionModel, self).__init__()
        self.model = model.to(device)
        self.device = device

    def train_step(self, optimizer, criterion, data_loader):
        """
        Performs a single training step, including forward pass, loss
        computation, and backpropagation for the entire dataset provided
        by the data loader.

        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer used to update the model's weights.
        criterion : nn.Module
            The loss function used to compute the loss.
        data_loader : torch.utils.data.DataLoader
            The data loader providing the training data.

        Returns:
        --------
        float
            The average training loss over the entire dataset.
        """
        self.model.train()
        train_loss = 0
        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(X.view(-1, 1, 128, 128))

            loss = criterion(output, y)
            loss.backward()
            train_loss += loss
            optimizer.step()

        return train_loss/len(data_loader.dataset)

    def validate(self, criterion, data_loader):
        """
        Evaluates the model on a validation dataset without updating model
        weights.

        Parameters:
        -----------
        criterion : nn.Module
            The loss function used to compute the loss.
        data_loader : torch.utils.data.DataLoader
            The data loader providing the validation data.

        Returns:
        --------
        float
            The average validation loss over the entire dataset.
        """
        self.model.eval()
        validation_loss = 0
        for X, y in data_loader:
            with torch.no_grad():
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X.view(-1, 1, 128, 128))
                loss = criterion(output, y)
                validation_loss += loss

        return validation_loss/len(data_loader.dataset)

    def train_model(self, tdataloader, vdataloader, epochs=50):
        """
        Trains the model over a specified number of epochs, and evaluates
        it on a validation dataset after each epoch.

        Parameters:
        -----------
        tdataloader : torch.utils.data.DataLoader
            The data loader providing the training data.
        vdataloader : torch.utils.data.DataLoader
            The data loader providing the validation data.
        epochs : int, optional, default=50
            The number of epochs to train the model.

        Returns:
        --------
        tuple
            A tuple containing two lists: the training loss and validation loss
            for each epoch.
        """
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        train_loss = []
        validation_loss = []
        for epoch in range(epochs):
            tloss = self.train_step(optimizer, criterion, tdataloader)
            vloss = self.validate(criterion, vdataloader)
            train_loss.append(tloss)
            validation_loss.append(vloss)
            tl = tloss.item()
            vl = vloss.item()
            print(f"Epoch [{epoch + 1}/{epochs}],",
                  f"Train Loss: {tl:.4f}, Val loss: {vl:.4f}")
        return train_loss, validation_loss

    def save_model(self, path):
        """
        Saves the model's state dictionary to a specified file.

        Parameters:
        -----------
        path : str
            The file path where the model's state dictionary will be saved.

        Returns:
        --------
        None
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Loads the model's state dictionary from a specified file.

        Parameters:
        -----------
        path : str
            The file path from which to load the model's state dictionary.

        Returns:
        --------
        None
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")
