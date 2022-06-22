import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os


class Actor(nn.Module):
    def __init__(self, input_dims, lr=0.01, cpt_dir='./', device=True, name='actor'):
        super(Actor, self).__init__()

        n_feats, n_actions, n_window = input_dims if len(input_dims) == 3 else input_dims[1:]
        self.cpt_dir = cpt_dir
        self.cpt_file = os.path.join(self.cpt_dir, name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # device

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=(1, 3), padding='same'),
            nn.Tanh()
        )

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=(1, n_window)),
            nn.Tanh()
        )

        # Third layer
        self.actions_prev = F.softmax(torch.randn((1, 1, n_actions, 1), device=self.device), dim=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats+1, out_channels=1, kernel_size=(1,1))
        )

        # Adding cash bias
        self.cash_bias = torch.ones([1]*len(self.actions_prev.shape), device=self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)
        x = torch.cat((x, self.actions_prev), dim=-3) # Concatenate actions from previous timestep

        x = self.conv3(x)
        x = torch.cat((self.cash_bias, x), dim=-2) # Add cash bias
        x = F.softmax(x, dim=-2)

        self.actions_prev = x[..., 1:, :] # Save actions to be taken (without cash bias)
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.cpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.cpt_file))


class Critic(nn.Module):
    def __init__(self, input_dims, lr=0.01, cpt_dir='./', name='critic'):
        super(Critic, self).__init__()

        n_feats, n_actions, n_window = input_dims if len(input_dims) == 3 else input_dims[1:]
        self.cpt_dir = cpt_dir
        self.cpt_file = os.path.join(self.cpt_dir, name)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # device

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=1, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # Second layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_window, out_channels=1, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # Third layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, n_actions))
        )

        # Fourth layer (Dense)
        self.linear = nn.Linear(in_features=1, out_features=1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = x.transpose(-3, -1)

        x = self.conv2(x)
        x = x.transpose(-2, -1)

        x = self.conv3(x)

        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.cpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.cpt_file))
