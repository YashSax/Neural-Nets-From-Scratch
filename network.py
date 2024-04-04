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
    
    def backward(self):
        reversed_modules = self.modules[::-1]

        loss_grad = self.criterion.backward()
        reversed_modules[0].backward(loss_grad)
        
        for module, next_module in zip(reversed_modules[1:], reversed_modules):
            module.backward(next_module.grad)

class MSELoss(LossFn):
    def __call__(self, y_true: np.array, y_pred: np.array):
        assert len(y_true.shape) == 1, "y_true must be 1-d"
        assert len(y_pred.shape) == 1, "y_pred must be 1-d"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape!"

        self.y_true = y_true
        self.y_pred = y_pred
        return np.sum((y_true - y_pred) ** 2) / len(y_pred)

    def backward(self):
        return 2 / len(self.y_pred) * (y_true - y_pred)
    

class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        in_dim_with_bias = self.in_dim + 1
        self.weights = np.random.normal(0, 1, (in_dim_with_bias, out_dim))

        self.layer_in = None
        self.layer_out = None

    def forward(self, x: np.array):
        x = np.append(x, [1])
        self.layer_in = x
        self.layer_out = x @ self.weights
        return self.layer_out


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
model.backward()
