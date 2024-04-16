import numpy as np
from typing import List
from .modules import *
from .loss_functions import *
from .optimizers import *


class Sequential():
    def __init__(self, modules: List[Module], criterion: LossFn, optimizer: Optimizer):
        self.modules = modules
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer.add_layers(modules)
        self.loss = 0
    
    def forward(self, x: np.array):
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def calculate_loss(self, y_true: np.array, y_pred: np.array):
        self.loss = self.criterion(y_true, y_pred)
        return self.loss
    
    def backward(self):
        reversed_modules = self.modules[::-1]

        prev_grad = self.criterion.backward()
        for module in reversed_modules:
            prev_grad = module.backward(prev_grad)
