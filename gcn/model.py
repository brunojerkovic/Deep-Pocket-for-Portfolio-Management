import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn.conv import ChebConv

import os


class GNNStack(nn.Module):
    def __init__(self, dim, K, num_layers, lr, dtype, device, cpt_dir, name):
        super(GNNStack, self).__init__()
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device
        self.cpt_dir = cpt_dir
        self.cpt_file = os.path.join(self.cpt_dir, name)

        # Conv layers
        self.convs = nn.ModuleList()
        for l in range(self.num_layers):
            self.convs.append(ChebConv(dim, dim, K, normalization=None))

        # Set optimizer and device to train the GNN network
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self._fix_laplacian(edge_attr, x) # TODO: use a hook for this
        #print(next(self.convs[0].parameters())._version)

        # TODO: uncomment this
        #emb = None
        #for i in range(self.num_layers):
        #    x = self.convs[i](x, edge_index, edge_attr)
        #    #emb = x
        #    x = torch.sigmoid(x)

        return x

    def loss(self, pred, label):
        return F.binary_cross_entropy(pred, label)

    def _fix_laplacian(self, edge_attr, x):
        """Get adjusted Laplacian matrix that conforms with Pytorch's implementation of ChebConv layer"""
        node_num = x.shape[0]
        # L = np.zeros((node_num, node_num))
        L = edge_attr.t()[0].reshape((node_num, node_num))
        D = torch.diag(torch.full((node_num,), node_num, dtype=self.dtype, device=self.device))
        L_ = -1 * L - D

        # Get new edge index
        edge_attrs = torch.unsqueeze(torch.ravel(L_), 1)
        return edge_attrs

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        torch.save(self.state_dict(), self.cpt_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.cpt_file))