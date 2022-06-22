from matplotlib import pyplot as plt
import numpy as np


def plot_losses(train_losses, val_losses, test_loss, plots_dir, restricted_ae, cut_plot=True):
    plt.title('Train/Validation/Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if cut_plot:
        plt.axis([-1, len(train_losses) + 1, -0.005, train_losses[1]])

    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, color='blue', label='train_loss')
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, color='orange', label='val_loss')
    plt.scatter(len(val_losses), test_loss, color='red', label='test_loss')
    plt.legend()

    plt_name = plots_dir + 'restricted.png' if restricted_ae else plots_dir + 'not_restricted.png'
    plt.savefig(plt_name)