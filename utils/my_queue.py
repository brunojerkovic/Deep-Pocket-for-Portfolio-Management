import torch
import numpy as np
from collections import deque


class MyQueue(deque):
    def __init__(self, maxlen, device, dtype):
        super(MyQueue, self).__init__(maxlen=maxlen)
        self.device = device
        self.dtype = dtype

    def full(self):
        return True if len(self) == self.maxlen else False

    def empty(self):
        return True if len(self) == 0 else False

    def append(self, element):
        popped_elements = self.popleft() if self.full() else None
        super().append(element)
        return popped_elements

    def pop(self):
        return self.popleft()

    def push(self, element):
        return self.append(element)

    def tensor(self):
        s = torch.stack(list(self)).unsqueeze(dim=0) #.clone()
        s.retain_grad()
        return s

    def tonumpy(self):
        return self.tensor().detach().cpu().numpy()

    def size(self):
        return len(self)
