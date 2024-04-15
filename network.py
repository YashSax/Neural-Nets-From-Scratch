import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Module(ABC):
    @abstractmethod
    def forward(self, x: np.array):
        pass

    def __call__(self, x: np.array):
        return self.forward(x)


class LossFn(ABC):
    @abstractmethod
    def __call__(self, y_true: np.array, y_pred: np.array):
        pass


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass


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


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = lr
    
    def add_layers(self, modules: List[Module]):
        self.layers = [mod for mod in modules if type(mod) == Layer]

    def step(self):
        for layer in self.layers:
            mean_grad = np.transpose(np.mean(layer.grad, axis=0))
            layer.weights -= self.lr * mean_grad


class Adam(Optimizer):
    def __init__(self, lr: float, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.ms = {}
        self.vs = {}
        self.epsilon = 1e-8
    
    def add_layers(self, modules: List[Module]):
        self.layers = [mod for mod in modules if type(mod) == Layer]
        for layer in self.layers:
            self.ms[layer] = np.zeros_like(layer.weights)
            self.vs[layer] = np.zeros_like(layer.weights)
    
    def step(self):
        for layer in self.layers:
            mean_grad = np.transpose(np.mean(layer.grad, axis=0))
            self.ms[layer] = self.beta1 * self.ms[layer] + (1 - self.beta1) * mean_grad
            self.vs[layer] = self.beta2 * self.vs[layer] + (1 - self.beta2) * (mean_grad ** 2)
            layer.weights -= self.lr * self.ms[layer] / np.sqrt(self.vs[layer] + self.epsilon)


class MSELoss(LossFn):
    def __call__(self, y_true: np.array, y_pred: np.array):
        assert len(y_true.shape) < 3 and len(y_pred.shape) < 3, "y_true and y_pred must be either 1-D or 2-D!"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape!"

        self.y_true = y_true
        self.y_pred = y_pred
        return np.sum((y_true - y_pred) ** 2) / np.prod(y_true.shape)

    def backward(self):
        self.grad = 2 / self.y_pred.shape[0] * (self.y_pred - self.y_true)
        return self.grad


class Layer(Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim
        in_dim_with_bias = self.in_dim + 1
        self.weights = np.random.normal(0, 1, (out_dim, in_dim_with_bias))

        self.layer_in = None
        self.layer_out = None

    def forward(self, x: np.array):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.layer_in = x
        self.layer_out = self.weights @ np.transpose(x)

        return np.transpose(self.layer_out)
    
    def backward(self, next_grad: np.array):
        layer_in_matrix = np.expand_dims(self.layer_in, axis=2)
        next_grad_matrix = np.expand_dims(next_grad, axis=1)
        self.grad = layer_in_matrix @ next_grad_matrix
        out = next_grad @ self.weights
        out = np.delete(out, -1, 1) # Remove bias grad
        return out


class ReLU(Module):
    def __init__(self):
        self.relu_in = None
    
    def forward(self, x: np.array):
        self.relu_in = x
        return np.maximum(x, np.zeros_like(x))
    
    def backward(self, next_grad: np.array):
        if self.relu_in is None:
            raise Exception("'forward' method has not been called on ReLU!")
        relu_grad = np.where(self.relu_in < 0, 0, 1)
        self.grad = np.multiply(relu_grad, next_grad)
        return self.grad


class Softmax(Module):
    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def backward(self, next_grad):
        raise NotImplementedError()