import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os


class MLP(nn.Module):
    def __init__(self, input_dims, lr, cpt_dir, device, name, weight_init_method):
        super(MLP, self).__init__()

        n_feats, n_actions, n_window = input_dims if len(input_dims) == 3 else input_dims[1:]
        self.cpt_dir = cpt_dir
        self.cpt_file = os.path.join(self.cpt_dir, name)
        self.device = device

        # First layer
        self.actor_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=(1, 3), padding='same'),
            nn.Tanh()
        )

        # Second layer
        self.actor_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=(1, n_window)),
            nn.Tanh()
        )

        # Third layer
        self.actions_prev = F.softmax(torch.randn((1, 1, n_actions, 1), device=self.device), dim=2)
        self.actor_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=n_feats+1, out_channels=1, kernel_size=(1,1))
        )

        # Adding cash bias
        self.cash_bias = torch.ones([1]*len(self.actions_prev.shape), device=self.device)

        # Set up the optimizer and device
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

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

    def loss(self, x):
        return -1 * x

    def forward(self, x):
        x = self.actor_conv1(x)

        x = self.actor_conv2(x)
        x = torch.cat((x, self.actions_prev), dim=-3) # Concatenate actions from previous timestep

        x = self.actor_conv3(x)
        x = torch.cat((self.cash_bias, x), dim=-2) # Add cash bias
        x = F.softmax(x, dim=-2)

        self.actions_prev = x[..., 1:, :].clone().detach() # Save actions to be taken (without cash bias)
        return x

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.cpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.cpt_file))