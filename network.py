import numpy as np
from abc import ABC, abstractmethod
from typing import List

# Create ABC for module, must have forward and grad methods
class Module(ABC):
    @abstractmethod
    def forward(self, x: np.array):
        pass

class LossFn(ABC):
    @abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array):
        pass

class Sequential():
    def __init__(self, modules: List[Module], criterion: LossFn):
        self.modules = modules
        self.criterion = criterion
        self.loss = 0
    
    def forward(self, x: np.array):
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def calculate_loss(self, y_true: np.array, y_pred: np.array):
        self.loss = self.criterion(y_true, y_pred)
        return self.loss

class MSELoss(LossFn):
    def __call__(self, y_true: np.array, y_pred: np.array):
        return np.sum(np.sqrt(y_true ** 2 + y_pred ** 2))

class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        in_dim_with_bias = self.in_dim + 1
        self.weights = np.random.normal(0, 1, (in_dim_with_bias, out_dim))

    def forward(self, x: np.array):
        x = np.append(x, [1])
        print(x.shape, self.weights.shape)
        return x @ self.weights

class ReLU(Module):
    def forward(self, x: np.array):
        return np.maximum(x, np.zeros_like(x))

class Softmax(Module):
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

# Let's write a simple model that predicts its own input
model = Sequential(
    [
        Layer(2, 10),
        ReLU(),
        Layer(10, 10),
        ReLU(),
        Layer(10, 2)
    ],
    criterion=MSELoss()
)


ex_in = np.array([1, 2])
y_pred = model.forward(ex_in)

y_true = ex_in
loss = model.calculate_loss(y_true, y_pred)

