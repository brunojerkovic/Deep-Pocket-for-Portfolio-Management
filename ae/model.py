import torch
import torch.nn as nn
import os


class Autoencoder(nn.Module):
    def __init__(self, checkpoint_dir, sizes, load_model=False):
        super(Autoencoder, self).__init__()
        self.input_size = sizes[0]
        self.hidden_size = sizes[3]
        self.output_size = sizes[-1]
        self.checkpoint_file = os.path.join(checkpoint_dir, 'AE')

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, sizes[1]),
            nn.Linear(sizes[1], sizes[2]),
            nn.Linear(sizes[2], self.hidden_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, sizes[4]),
            nn.Linear(sizes[4], sizes[5]),
            nn.Linear(sizes[5], self.output_size)
        )

        # Load model if needed
        if load_model:
            self.load_checkpoint()

        self.loss = nn.MSELoss()

    def forward(self, x):
        latent_repr = self.encoder(x)
        output = self.decoder(latent_repr)
        return output

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
