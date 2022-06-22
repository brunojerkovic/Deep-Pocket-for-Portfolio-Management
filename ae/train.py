import torch
import torch.nn as nn
import numpy as np
from ae.model import Autoencoder
from ae.result_saver import plot_losses
import wandb


def validation(model, val_loader, criterion):
    model.eval()  # Model acts in eval mode
    with torch.no_grad():  # Disable saving gradients
        val_losses_curr = []
        for i_batch, y_true in enumerate(val_loader):
            y_true = y_true.float()

            # Forward pass
            y_pred = model(y_true)
            y_true = y_true[:, :y_pred.shape[1]]

            # Compute and print loss
            loss = criterion(y_pred, y_true)
            val_losses_curr.append(loss.item())

    model.train()  # Model acts in train mode (again)
    return sum(val_losses_curr) / len(val_losses_curr)


def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, result_saver, patience=None, verbose=True):
    train_losses = []
    val_losses = []
    trigger_times = 0
    min_val_loss = np.inf

    for epoch in range(epochs):
        train_losses_curr = []
        for i_batch, y_true in enumerate(train_loader):
            y_true = y_true.float()

            # Forward pass
            y_pred = model(y_true)
            y_true = y_true[:, :y_pred.shape[1]]

            # Compute and print loss
            loss = criterion(y_pred, y_true)
            train_losses_curr.append(loss.item())

            # Zero (current) gradients, perform a backward pass (calc gradients), and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate losses from this epoch
        train_loss_epoch = sum(train_losses_curr) / len(train_losses_curr)
        val_loss_epoch = validation(model, val_loader, criterion)
        result_saver.log_info({'train_loss': train_loss_epoch})
        result_saver.log_info({'val_loss': val_loss_epoch})

        # Early stopping
        if patience:
            if val_loss_epoch > min_val_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f"Model is stopped on epoch {epoch}")
                    return train_losses, val_losses
            else:
                min_val_loss = val_loss_epoch
                trigger_times = 0

        # Print (and save) losses from this epoch
        if verbose:
            train_info = f"Epoch: {epoch + 1}/{epochs} | Train Loss: {train_loss_epoch} | Val Loss: {val_loss_epoch}"
            if trigger_times:
                train_info += f"| Min Loss: {min_val_loss} | Trigger: {trigger_times}"
            print(train_info)
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)

        # Save model parameters
        model.save_checkpoint()

        # Stream to wandb
        result_saver.stream(idx=epoch)

    return train_losses, val_losses


def train(data_loaders, epochs, patience,
          chkpt_dir, plots_dir, device,
          restricted_ae, save_plots, result_saver=None):
    # Get ae loaders
    test_loader = None
    if len(data_loaders) == 3:
        train_loader, valid_loader, test_loader = data_loaders
    else:
        train_loader, valid_loader = data_loaders

    # Instantiate model, loss, optimizer
    model = Autoencoder(chkpt_dir, [11, 10, 9, 3, 3, 3, 3]) if restricted_ae else Autoencoder(chkpt_dir,
                                                                                              [11, 10, 9, 3, 9, 10, 11])
    model.to(device)
    criterion = model.loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result_saver.watch_models(model)

    # Train, validate and test model
    train_losses, val_losses = train_loop(model, train_loader, valid_loader, criterion, optimizer, epochs, result_saver, patience)

    if test_loader is not None:
        test_loss = validation(model, test_loader, criterion)
        print(f"Test Loss: {test_loss}")

    # Plot losses
    if save_plots:
        if test_loader is not None:
            plot_losses(train_losses, val_losses, test_loss, plots_dir, restricted_ae)
        else:
            plot_losses(train_losses, val_losses, 0., plots_dir, restricted_ae)
