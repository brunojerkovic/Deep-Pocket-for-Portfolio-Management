import torch
import torch.nn as nn
from torch import optim
import os


class Critic(nn.Module):
    def __init__(self, input_dims, lr, cpt_dir, device, name, weight_init_method):
        super(Critic, self).__init__()

        n_feats, n_actions, n_window = input_dims if len(input_dims) == 3 else input_dims[1:]
        self.cpt_dir = cpt_dir
        self.cpt_file = os.path.join(self.cpt_dir, name)
        self.device = device

        # First layer
        self.critic_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # Second layer
        self.critic_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_window, out_channels=1, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # Third layer
        self.critic_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, n_actions))
        )

        # Fourth layer (Dense)
        self.linear = nn.Linear(in_features=1, out_features=1)

        # Set up the loss, optimizer and the device
        self.loss = nn.MSELoss()
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize weights
        self.weight_init_method = weight_init_method
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        # Set up parameters
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if self.weight_init_method == 'uniform':
                nn.init.uniform_(module.weight.data)
            elif self.weight_init_method == 'xavier':
                nn.init.xavier_uniform_(module.weight.data)
            else:
                nn.init.normal_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.critic_conv1(x)
        x = x.transpose(-3, -1)

        x = self.critic_conv2(x)
        x = x.transpose(-2, -1)

        x = self.critic_conv3(x)

        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.cpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.cpt_file))
