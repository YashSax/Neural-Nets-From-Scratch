import numpy as np
from abc import ABC, abstractmethod
from typing import List

# Create ABC for module, must have forward and grad methods
class Module(ABC):
    @abstractmethod
    def forward(self, x: np.array):
        pass

class Sequential():
    def __init__(self, modules: List[Module]):
        self.modules = modules
    
    def forward(self, x: np.array):
        for module in self.modules:
            print(x)
            x = module.forward(x)
        return x

class MSELoss():
    def __call__(self, y_true, y_pred):
        return np.sum(np.sqrt(y_true ** 2 + y_pred ** 2))

class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int, weights=None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad = np.zeros_like(weights)

        self.weights = weights
        if self.weights is None:
            self.weights = np.random.normal(0, 1, (in_dim, out_dim))

    def forward(self, x: np.array):
        return x @ self.weights

class ReLU(Module):
    def forward(self, x: np.array):
        return np.maximum(x, np.zeros_like(x))

class Softmax(Module):
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

model = Sequential([
    Layer(2, 10),
    ReLU(),
    Layer(10, 10),
    ReLU(),
    Layer(10, 2),
    Softmax()
])

ex_in = np.array([1, 2])
ex_out = model.forward(ex_in)

criterion = MSELoss()
