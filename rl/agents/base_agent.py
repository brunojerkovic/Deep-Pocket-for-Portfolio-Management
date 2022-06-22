from abc import ABC, abstractmethod
import torch.nn as nn


class Agent(ABC):
    def __init__(self, gamma, lr_a, lr_c, weight_init_method, input_dims, batch_size,
                 cpt_dir, device, buffer_size):
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.cpt_dir = cpt_dir
        self.device = device
        self.weight_init_method = weight_init_method

        self.log_actions = None


    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self, state, reward, next_state, done, gcn):
        pass

    @abstractmethod
    def save_models(self):
        pass

    @abstractmethod
    def load_models(self):
        pass

    @abstractmethod
    def get_models(self) -> list:
        pass
