import torch
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os

from datasets.gcn.dataset import GraphDataset
from gcn import GNNStack
from utils.my_queue import MyQueue


class StockEnvironment(gym.Env):
    def __init__(self, env_folderpath, reward, buffer_size, graph_name, stock_dataset, stocknames_config_filepath,
                 gnn_K, gnn_num_layers, gnn_lr, period,
                 gnn_cpt_dir, device):
        self.env_folderpath = env_folderpath
        self.buffer_size = buffer_size
        self.device = device
        self.stock_dataset = stock_dataset

        self.graph_dataset = GraphDataset(root=os.path.join(env_folderpath, 'input_data'),
                                          n=buffer_size,
                                          stock_dataset=stock_dataset,
                                          graph_name=graph_name,
                                          stocknames_config_filepath=stocknames_config_filepath,
                                          dtype=torch.float32,
                                          device=device)
        self.reward = reward
        self.gcn = GNNStack(dim=self.graph_dataset.n_feats,
                            K=gnn_K,
                            num_layers=gnn_num_layers,
                            lr=gnn_lr,
                            dtype=torch.float32,
                            device=device,
                            cpt_dir=gnn_cpt_dir,
                            name='GCN')

        self.action_space = Discrete(self.graph_dataset.node_num + 1)
        highest_obs = np.array([[1e3]*self.graph_dataset.n_feats]*self.graph_dataset.node_num)
        lowest_obs = np.array([[-1e3] * self.graph_dataset.n_feats] * self.graph_dataset.node_num)
        self.observation_space = Box(low=lowest_obs, high=highest_obs)

        #self.n_steps = 50 - buffer_size # 4553 # TODO: (FOR FULL IMPL) len(self.graph_dataset) - buffer_size

        # Initialize counter start and end
        self.start_idx = self.stock_dataset.date_to_idx(period[0])
        self.end_idx = self.stock_dataset.date_to_idx(period[1])
        self.n_steps = self.end_idx - self.start_idx - buffer_size

        self.iter = self.init_iter = buffer_size + self.start_idx

        # Fill up buffer to get the first state
        self.buffer = MyQueue(maxlen=buffer_size,
                              dtype=torch.float32,
                              device=device)

        # Fill up the buffer for the first state
        while not self.buffer.full():
            graph_data = self.graph_dataset[self.iter]
            convolved_graph = self.gcn(graph_data)
            self.buffer.push(convolved_graph)
            self.iter += 1
        self.state = self.buffer.tensor()

    def step(self, action):
        # Perform optimization of GCN you are not in the first iteration
        if self.iter - self.buffer.size() > self.init_iter:
            # TODO: (FOR FULL IMPL) uncomment this
            #self.gcn.optimizer.step()
            #self.gcn.optimizer.zero_grad()
            pass

        # Get reward from being in this state
        actions = action.clone().squeeze().detach().cpu().numpy()  # Reshape actions (and turn to numpy)
        reward = self.reward.get_reward(state_idx=self.iter, actions=actions)

        # GCN forward pass
        graph_data = self.graph_dataset[self.iter]
        convolved_graph = self.gcn(graph_data)
        self.buffer.push(convolved_graph)
        self.state = self.buffer.tensor()

        # Get info about the state
        info = {}

        # Move iter for the next state
        self.iter += 1

        # Check if episode is done
        done = True if self.iter == self.n_steps else False

        return self.state, reward, done, info

    def render(self):
        # Implement visualization stuff (not needed for now)
        super(StockEnvironment, self).render()

    def reset(self):
        self.iter = self.init_iter # Reset counter

        # Fill up buffer to get the first state
        self.buffer = MyQueue(maxlen=self.buffer_size,
                              dtype=torch.float32,
                              device=self.device)

        # Fill up the buffer for the first state
        while not self.buffer.full():
            graph_data = self.graph_dataset[self.iter]
            convolved_graph = self.gcn(graph_data)
            self.buffer.push(convolved_graph)
            self.iter += 1
        self.state = self.buffer.tensor()

        return self.state

    def save_gcn_model(self):
        self.gcn.save_checkpoint()

