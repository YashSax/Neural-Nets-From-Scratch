import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, x: np.array):
        pass

    def __call__(self, x: np.array):
        return self.forward(x)

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

class Sigmoid(Module):
    def __init__(self):
        self.sigmoid_out = None
    
    def forward(self, x: np.array):
        self.sigmoid_out = 1 / (1 + np.exp(-1 * x))
        return self.sigmoid_out
    
    def backward(self, next_grad: np.array):
        if self.sigmoid_out is None:
            raise Exception("'forward' method has not been called on Sigmoid!")
        relu_grad = np.multiply(self.sigmoid_out, 1 - self.sigmoid_out)
        self.grad = np.multiply(relu_grad, next_grad)
        return self.grad
